"""M2 视觉理解层（方案 A 核心）：把 Footage Index 从"元数据级"升级为"可检索级"。

职责（对应 PRD §7.1 / §7.2）：
- YOLOv8 对关键帧做物体/人物检测，跨关键帧合并为带时间戳的视觉段（visual_segments）；
- FunASR(Paraformer) 对音轨转写，产出带时间戳台词段（transcript_segments）；
- 二者回填 Footage Index，使 Agent 的 Retriever 能按"猫 / 人脸 / 发言人 / 台词"检索到片段，
  从而把"剪掉有猫那段"映射为具体的 sourceIn/sourceOut。

实现选型说明（主理人已确认方案 A）：
- 原 PRD 写"YOLOv8 + MediaPipe 互补"；但 MediaPipe 在 Python 3.13 上大概率无官方 wheel，
  故人脸/姿态统一改用 YOLOv8 系列模型（yolov8n-face / yolov8n-pose.pt），不引入 MediaPipe 依赖。
- 所有重依赖（ultralytics / funasr / torch）均惰性 import，保证本模块在依赖未装时仍可被 import 检查。

注意：本文件为 M2 骨架，FunASR 返回结构与模型 revision 需在依赖落地后按真实输出校准。
"""
from __future__ import annotations

import os
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from .footage_index import FootageIndex, VisualSegment, TranscriptSegment
from .ingest import extract_frames, ffmpeg_bin_dir


# ---------------------------------------------------------------------------
# 模型缓存（同一进程内只加载一次）
# ---------------------------------------------------------------------------
# 为什么不每次重新构造：YOLO 载权约 1–3s，Paraformer 载权约 10–20s，且 AutoModel
# 默认会联网做「版本检查」，弱网时可能长时间阻塞（M5 界面里表现为「点了没反应」）。
# 缓存后：第二次点击 M2 快 5–10 倍，且不再依赖网络。
_YOLO_CACHE: Dict[str, Any] = {}
_ASR_CACHE: Dict[Tuple[str, str, str], Any] = {}
_VAD_CACHE: Dict[str, Any] = {}

# VAD 静音阈值（毫秒）：>800 会漏切（幕间停顿被吞），300 实测能把幕分干净。
_VAD_END_SIL_MS = 300

# 转写统一采样率：抽音频与内存切片都按它换算（16k 是 Paraformer 的输入要求）。
_SR = 16000


def _get_yolo(weights: str):
    if weights not in _YOLO_CACHE:
        from ultralytics import YOLO
        _YOLO_CACHE[weights] = YOLO(weights)
    return _YOLO_CACHE[weights]


def _get_asr(model: str, model_revision: str, device: str):
    key = (model, model_revision, device)
    if key not in _ASR_CACHE:
        from funasr import AutoModel
        # disable_update：跳过联网版本检查（离线也会卡的地方）；disable_pbar：静默进度条
        _ASR_CACHE[key] = AutoModel(model=model, model_revision=model_revision,
                                    device=device, disable_update=True,
                                    disable_pbar=True)
    return _ASR_CACHE[key]


def _get_vad(device: str):
    """fsmn-vad 声学模型（约 6MB，实测存在 D:\\Dev\\Cache\\modelscope）。

    单独实例化它（而不是挂到 AutoModel(vad_model=...)）才能拿到**原始语音段边界** ——
    挂进 AutoModel 时它只是内部切段后又把结果拼成一条，段边界就丢了。

    `max_end_silence_time=300`（默认 800）是**实测调出来的**：
    - 默认 800ms：素材2 六幕旁白只切出 **3 段**（幕间停顿短于 800ms 被合并），
      于是"删掉说第三幕的"会连带第四幕一起删；
    - 300ms：切出 **11 段**，段边界落在 `4150/8150/12140/16160/20160`，
      与素材的幕边界 `4/8/12/16/20s` **精确吻合**（差值即旁白起始的 0.25s 延时）。
    注意：调细之后**纯音乐素材也会被切出多段**（素材3 → 6 段），所以护栏必须按
    "整片是否像无语音素材"来判断，不能只看单段覆盖率 —— 见 `_looks_like_hallucination`。

    参数**直接作为关键字传**（`VADXOptions(**kwargs)` 消费，见 fsmn_vad_streaming/model.py:397）；
    签名里的 `vad_post_args` 虽存在但从未被使用 —— 传它会静默无效（踩过）。
    """
    if device not in _VAD_CACHE:
        from funasr import AutoModel
        _VAD_CACHE[device] = AutoModel(model="fsmn-vad", device=device,
                                       disable_update=True, disable_pbar=True,
                                       max_end_silence_time=_VAD_END_SIL_MS)
    return _VAD_CACHE[device]


# ---------------------------------------------------------------------------
# 时间合并工具：把离散关键帧上的同标签检测合并成连续段
# ---------------------------------------------------------------------------
def _merge_times(times: List[float], gap: float) -> List[Tuple[float, float]]:
    if not times:
        return []
    times = sorted(set(times))
    segs: List[Tuple[float, float]] = []
    s = prev = times[0]
    for t in times[1:]:
        if t - prev <= gap:
            prev = t
        else:
            segs.append((s, prev))
            s = prev = t
    segs.append((s, prev))
    return segs


_MIN_SEG = 0.5   # 无镜头信息时段的兜底最小长度：绝不能产出零长度区间


def _scene_index_at(scenes: List[Dict[str, Any]], t: float) -> Optional[int]:
    """时刻 t 属于第几个镜头。落在边界上归**下一镜** —— 与 ffmpeg 抽帧语义一致
    （`-ss 4.0` 取到的是 4.0 之后那一帧，即下一幕的画面）。"""
    idx: Optional[int] = None
    for k, sc in enumerate(scenes):
        if t >= float(sc["start"]) - 1e-6:
            idx = k
    return idx


def _label_ranges_from_times(scenes: List[Dict[str, Any]], times: List[float],
                             gap: float) -> List[Tuple[float, float]]:
    """把「命中关键帧」外扩成「命中镜头」，返回秒数区间列表。

    为什么必须外扩成镜头，而不是直接用关键帧跨度（两条都是实测踩出来的）：

    1) **语义删除删的是整个镜头。** 只按关键帧跨度删，会留下同一镜头里没被
       关键帧覆盖的残段。实测：源里 `0–5s` 的公交镜头只在 `0.5 / 2 / 3 / 4s`
       命中了关键帧 → 区间 `(0.5, 4.0)` → 删完成片里**仍留着 0–0.5s 与 4–5s
       的公交画面**，用户看到的还是"bus 没删干净"。
    2) **单关键帧命中的镜头会合并出零长度区间 `(t, t)`**（`_merge_times` 对单个
       时刻返回 `(t, t)`）。`delete_range` 要求 `start < end` → 严格模式直接报错；
       非严格模式（UI 走的就是这条）被**静默跳过**，又是一次"悄悄少删"。

    相邻镜头若都命中则合并成一段，因此"一整幕都是公交"会得到该幕的完整区间。
    """
    if not times:
        return []
    if not scenes:
        # 没有镜头信息时退化为关键帧跨度，但保证非零长度
        return [(s, max(e, s + _MIN_SEG)) for (s, e) in _merge_times(sorted(set(times)), gap)]

    hit = {i for i in (_scene_index_at(scenes, t) for t in sorted(set(times)))
           if i is not None}
    ranges: List[Tuple[float, float]] = []
    for i in sorted(hit):
        s, e = float(scenes[i]["start"]), float(scenes[i]["end"])
        if ranges and abs(ranges[-1][1] - s) < 1e-6:
            ranges[-1] = (ranges[-1][0], e)      # 相邻镜头命中 → 并成一段
        else:
            ranges.append((s, e))
    return ranges


# ---------------------------------------------------------------------------
# 1) YOLOv8 视觉检测
# ---------------------------------------------------------------------------
def detect_objects(index: FootageIndex,
                   weights: str = "yolov8n.pt",
                   conf: float = 0.3,
                   device: str = "cpu",
                   keyframe_dir: Optional[str] = None) -> List[VisualSegment]:
    """对关键帧跑 YOLOv8，把同标签跨关键帧合并为带时间戳的视觉段。

    - weights：默认 yolov8n.pt（COCO 80 类，含 person/cat/dog 等）；
      人脸/近景可换 yolov8n-face.pt，姿态可换 yolov8n-pose.pt。
    - 返回 VisualSegment 列表，source="yolo"。
    """
    import cv2  # noqa: F401（确保 opencv 可用）

    if not index.keyframes:
        return []

    tmp = keyframe_dir or tempfile.mkdtemp()
    frames = extract_frames(index.filepath, index.keyframes, tmp)
    model = _get_yolo(weights)

    # 关键帧时间 -> 该帧检测到的标签集合
    time_labels: Dict[float, set] = {}
    for t, fr in zip(index.keyframes, frames):
        res = model(fr, conf=conf, device=device, verbose=False)[0]
        labels = {model.names[int(b.cls)] for b in res.boxes}
        time_labels[t] = labels

    # 逐标签合并时间段：以**镜头**为单位（不是关键帧跨度），原因见 _label_ranges_from_times
    segs: List[VisualSegment] = []
    all_labels = set().union(*time_labels.values()) if time_labels else set()
    step = (index.keyframes[1] - index.keyframes[0]) if len(index.keyframes) > 1 else 2.0
    gap = 2.0 * step
    for lab in sorted(all_labels):
        ts = [t for t, labs in time_labels.items() if lab in labs]
        for (s, e) in _label_ranges_from_times(index.scenes, ts, gap):
            segs.append(VisualSegment(label=lab, start=s, end=e,
                                      confidence=conf, source="yolo"))
    return segs


# ---------------------------------------------------------------------------
# 2) FunASR 语音转写
# ---------------------------------------------------------------------------
# 下面是**实测**出来的 FunASR 行为，不要再凭直觉假设（2026-09-23 四次探测）：
#
# 1) `generate()` 的返回体只有 `{"key", "text"}` 两个键 —— **没有 `sentence_info`**，
#    `sentence_timestamp=True` 传了也不生效，返回体里也不含任何 `timestamp`。
#    结论：Paraformer 本体给不出时间戳。要句级断句得挂 `punc_model`（数百 MB，暂不引入）。
# 2) 把 `vad_model="fsmn-vad"` **挂在 AutoModel 里**也没用：它只是内部切段后又把结果
#    拼成一条返回，段边界被丢掉（实测仍 1 段、仍无 `timestamp`）。
#    → 所以本项目**单独实例化 VAD**（见 `_get_vad`），自己拿段边界。
# 3) VAD 段是**段落级**、不是句级：实测素材2 六幕旁白只切出 3 段
#    （`80–6710 / 8190–14530 / 16160–23980` ms）—— 幕间停顿短于 VAD 的静音阈值就被合并。
#    所以它给出的是"成块的"时间戳，比"整片一条"强得多，但别当作逐句。
# 4) **VAD 会把纯音乐判成语音**（实测素材3 纯音乐 → `[[0, 19980]]` 一整段覆盖全片）——
#    这就是"挂了 VAD 也治不了幻觉"的根因。→ 幻觉护栏必须保留，见
#    `_looks_like_hallucination()`。
def _vad_spans(wav: str, device: str, duration: float) -> List[Tuple[float, float]]:
    """用 fsmn-vad 切出语音段的**秒**区间。任何异常都退化为 []（调用方走整片兜底）。

    实测返回结构：`res[0]["value"] == [[beg_ms, end_ms], ...]`；未闭合的段 `end` 可能为
    `-1`，此时用素材时长兜底（不然会切出一个负长度的段）。
    """
    try:
        res = _get_vad(device).generate(input=wav)
    except Exception:                                       # noqa: BLE001
        return []                                           # 下不到权重/推理失败都不该阻断转写
    if not res:
        return []
    spans: List[Tuple[float, float]] = []
    for item in (res[0].get("value") or []):
        try:
            s, e = float(item[0]) / 1000.0, float(item[1]) / 1000.0
        except (TypeError, IndexError, ValueError):
            continue
        if e <= 0:                                          # -1 表示未闭合
            e = duration
        e = min(e, duration) if duration > 0 else e
        if e > s:
            spans.append((s, e))
    return spans


def _slice_wav(ff: str, src: str, start: float, end: float) -> str:
    """切出 `[start, end]` 的单声道 16k 临时 wav，返回路径（调用方负责删除）。

    `-ss/-to` 放在 `-i` **之前**（输入侧定位）。对 wav 这种无关键帧的格式同样精确，
    且比输出侧 `-ss` 快得多。
    """
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run([ff, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
                    "-ac", "1", "-ar", "16000", out],
                   capture_output=True, check=True)
    return out


def _union_len(spans: List[Tuple[float, float]]) -> float:
    """区间并集的总长度（段可能重叠，不能直接 sum）。"""
    if not spans:
        return 0.0
    total = 0.0
    cs: Optional[float] = None
    ce: Optional[float] = None
    for s, e in sorted(spans):
        if cs is None or ce is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)                      # 重叠 → 合并
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs) if cs is not None and ce is not None else total


# 整片都是 ≤ 这么多个字的短语 → 判定为"无语音素材被切细后逐段幻觉"
_SILENT_MAX_CHARS = 3


def _looks_like_silent_footage(cands: List[Tuple["TranscriptSegment", bool]],
                               duration: float) -> bool:
    """**整体判据**：这批段是不是"无语音素材被 VAD 切细 + 逐段幻觉"。

    为什么需要它（实测踩出来的）：把 VAD 静音阈值调到 300ms 后，20s **纯音乐**素材
    从 1 段变成 **6 段**，逐段转写出 `'嗯嗯' / '呜啦啦' / '呜呜呜' / '嗯' / '嗯' / '嗯'`。
    此时**单段判据 `_looks_like_hallucination` 全部漏过**：
      * 每段只覆盖 3–4s（≈16% 时长）→ `covers_all` 不成立；
      * 文本**含汉字**（'嗯' 也是 CJK）→ `has_cjk` 反而成了"看起来像真台词"的证据。

    判据（必须同时成立 —— 任一条都足以让它不可能是"整片真人语音"）：
      1. 段数 ≥ 3：多段才有"整体"可言；单段素材仍走 `covers_all` 老路。
      2. 每段文本都 ≤ `_SILENT_MAX_CHARS` 个字：真人不可能整片只说 1–3 字的短语
         （对照实测：素材2 的真旁白最长段 9 字，一眼可分）。
      3. 段区间并集覆盖 ≥ 90% 时长：说明这个素材**从头到尾"都在说话"**，
         而真语音素材总有明显静音留白。

    取舍：若素材真的整片只有 1–3 字短句（如全程"嗯""好"的录音），会被整体丢弃。
    概率极低，且转写段只进事实块/展示、**不参与删除决策**，故可接受。
    """
    if duration <= 0 or len(cands) < 3:
        return False
    segs = [seg for seg, _ in cands]
    if any(len((s.text or "").strip()) > _SILENT_MAX_CHARS for s in segs):
        return False
    covered = _union_len([(s.start, s.end) for s in segs])
    return covered >= duration * 0.9


def _read_wav_mono16k(path: str):
    """把 wav 读成 float32 单声道数组；不可用/采样率不符时返回 None。

    惰性 import soundfile：这是可选优化路径，缺它不该影响转写本身（调用方会退回 ffmpeg 切段）。
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except Exception:                                   # noqa: BLE001
        return None
    if int(sr) != _SR:
        return None
    if getattr(audio, "ndim", 1) > 1:                   # 保险：多声道求均值
        audio = audio.mean(axis=1)
    return audio


def _asr_segment_texts(am, wav: str, ff: str,
                       spans: List[Tuple[float, float]]) -> List[str]:
    """按 `spans` 逐段转写，返回与 spans **等长**的文本列表（失败项为空串）。

    优先走 **内存切片 + 批量转写**：实测 11 段素材从 5.02s 降到 1.36s ——
    因为原来 3.30s 全花在 11 次 ffmpeg 切音频的进程开销上，而内存切片只要 0.003s。
    批量 `generate(input=[...])` 也省掉了每次调用的前后处理重复。

    两级兜底（都不能少）：
      1. soundfile 不可用 / 采样率不符 / 切片后段数对不上 → 退回 ffmpeg 切段；
      2. 批量返回的**条数**与输入不一致 → 也退回逐条（否则时间戳会错位到别的段，
         这比慢一点严重得多）。
    """
    audio = _read_wav_mono16k(wav)
    if audio is not None:
        chunks = [audio[int(s * _SR):int(e * _SR)] for (s, e) in spans]
        if len(chunks) == len(spans) and all(len(c) > 0 for c in chunks):
            try:
                raw = am.generate(input=chunks, batch_size_s=300, return_spk_res=False)
                if len(raw) == len(chunks):
                    return [((r.get("text") or "").strip()) for r in raw]
            except Exception:                           # noqa: BLE001
                pass                                    # 交给下面的兜底

    texts: List[str] = []
    for (s, e) in spans:
        piece = None
        try:
            piece = _slice_wav(ff, wav, s, e)
            raw = am.generate(input=piece, batch_size_s=300, return_spk_res=False)
            texts.append("".join((r.get("text") or "") for r in raw).strip())
        except Exception:                               # noqa: BLE001
            texts.append("")                            # 单段失败不拖垮整体
        finally:
            if piece:
                try:
                    os.remove(piece)
                except OSError:
                    pass
    return texts


def _looks_like_hallucination(seg: "TranscriptSegment", duration: float,
                              from_sentence_info: bool) -> bool:
    """判定一个转写段是否是"无语音素材上的幻觉"。

    护栏取值刻意保守 —— **只在结构上不可能成立时才丢**：
      * 来自真实断句（`sentence_info`）的一律保留，不管内容长短；
      * 必须同时满足「覆盖近全片」+「不含汉字」+「词数 ≤ 2」。
    这样 `'the'` / `'Thank you'`（纯音乐与静音的典型幻觉）会被丢掉，而
    素材2 那种中文旁白（含汉字、字数多）**不会**被误杀。

    注意 **VAD 段不算 `from_sentence_info`**：VAD 只是按静音切块，它会把纯音乐整段判成
    语音（实测），所以 VAD 段仍要走这三个条件过一遍 —— 这正是素材3 能被拦住的原因。

    取舍：万一素材里真的只有一句极短英文台词（如 "Go!"），也会被丢掉。
    影响可控 —— 转写段目前只进事实块与界面展示，**不参与删除决策**。
    """
    if from_sentence_info:
        return False
    text = (seg.text or "").strip()
    if not text:
        return True
    covers_all = (seg.end - seg.start) >= max(1.0, duration * 0.9)
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    tiny = len(text.split()) <= 2 and len(text) <= 12
    return covers_all and (not has_cjk) and tiny


def transcribe(index: FootageIndex,
               model: str = "paraformer",
               model_revision: str = "v2.0.4",
               device: str = "cpu",
               use_vad: bool = True) -> List[TranscriptSegment]:
    """ffmpeg 抽单声道 16k wav -> FunASR(Paraformer) 转写，返回带时间戳台词段。

    无音轨素材直接返回空列表。

    时间戳来源分两级（Paraformer **本体给不出句级时间戳**，见上方实测结论）：
      1. **VAD 分段**（默认，`use_vad=True`）：用 fsmn-vad 切出语音块，**逐块**转写 ——
         每块产出一个段，时间戳取 VAD 的块边界，文本顺序也天然正确（不再"整片一条"、
         不再出现"第四幕…第六幕…第五幕…"这种模型自己顺出来的乱序）。
         粒度是**段落级**（实测素材2 六幕旁白 → 11 段），比整片一条强得多，但不是逐句。
         逐块转写走**内存切片 + 批量调用**（实测 11 段 5.02s → 1.36s；见 `_asr_segment_texts`）。
      2. **整片兜底**：VAD 不可用（下不到权重 / 推理失败）时退化为整片一条，行为与旧版一致。

    两级结果都要过 `_looks_like_hallucination()` 护栏 —— VAD 会把纯音乐整段判成语音，
    所以"有 VAD"**不等于**"没幻觉"。
    """
    if not index.metadata.get("has_audio"):
        return []

    bin_dir = ffmpeg_bin_dir()
    ff = os.path.join(bin_dir, "ffmpeg") if bin_dir else "ffmpeg"
    wav = tempfile.mktemp(suffix=".wav")
    subprocess.run([ff, "-y", "-i", index.filepath, "-vn",
                    "-ac", "1", "-ar", "16000", wav],
                   capture_output=True, check=True)

    duration = float(index.metadata.get("duration", 0.0))
    # 记录每段的来源，用于幻觉判定（模型真实断句 vs VAD 块 / 整片兜底）
    cands: List[Tuple[TranscriptSegment, bool]] = []
    try:
        am = _get_asr(model, model_revision, device)
        spans = _vad_spans(wav, device, duration) if use_vad else []

        if spans:
            texts = _asr_segment_texts(am, wav, ff, spans)
            for (s, e), text in zip(spans, texts):
                if text:
                    cands.append((TranscriptSegment(start=s, end=e, text=text), False))
        else:
            raw = am.generate(input=wav, batch_size_s=300, return_spk_res=False)
            for r in raw:
                # 优先用 sentence_info（含逐句起止）；否则退化为整段一条
                if r.get("sentence_info"):
                    for sent in r["sentence_info"]:
                        cands.append((TranscriptSegment(
                            start=float(sent["start"]), end=float(sent["end"]),
                            text=sent["text"]), True))
                elif r.get("text"):
                    cands.append((TranscriptSegment(
                        start=0.0, end=index.metadata["duration"],
                        text=r["text"]), False))
    finally:
        try:
            os.remove(wav)                       # 临时 wav 用完即删，避免堆积
        except OSError:
            pass

    # 整批判据优先：命中说明整个素材是"无语音 + 被 VAD 切细后逐段幻觉"，
    # 此时单段判据会全部漏过（每段覆盖率都不高、文本还含汉字），必须整体丢弃。
    if _looks_like_silent_footage(cands, duration):
        return []
    return [seg for seg, from_sentence in cands
            if not _looks_like_hallucination(seg, duration, from_sentence)]


# ---------------------------------------------------------------------------
# 3) M2 主编排：回填 Footage Index
# ---------------------------------------------------------------------------
def build_visual_understanding(index: FootageIndex, **kw) -> FootageIndex:
    """填充 index.visual_segments 与 index.transcript_segments，返回 enriched index。

    kw 可传：weights / conf / device（视觉），model / model_revision / device（语音）。
    """
    detect_kw = {k: kw[k] for k in ("weights", "conf", "device") if k in kw}
    asr_kw = {k: kw[k] for k in ("model", "model_revision", "device") if k in kw}

    index.visual_segments = detect_objects(index, **detect_kw)
    index.transcript_segments = transcribe(index, **asr_kw)
    return index
