"""M4 编辑智能体：自然语言 → 结构化 Edit Operation。

职责（对应 PRD §7 / M4）：
- 把用户的中文/英文自然语言剪辑指令，映射为 M3 的 {type, payload} 操作列表；
- 确定性编译器(operations.apply_operation)消费这些操作，保证"AI 只发结构化操作"铁律；
- 语义检索："删掉有 bus 的片段" 通过 FootageIndex.visual_segments 反查时间区间 → 操作。

实现选型（主理人已确认 M4 走轻量自定义，不引入 LangGraph 重依赖）：
- 默认实现 = RuleBasedPlanner（离线、确定性、可单测），作为无 LLM Key 时的可用降级；
- 真实 LLM（DeepSeek / OpenAI / 本地 Qwen）通过 LLMProvider 接口热插拔，见 JsonPlanner。

LLM 路径的两处工程加固（M5 实跑踩坑后补，2026-09-23）：
1. **上下文**：JsonPlanner 现在把「Timeline 可用 clip id 清单」与「M2 素材理解事实
   （镜头 / 检出物体时间区间 / 台词）」一起塞进提示词。此前只给 timeline JSON，
   模型会臆造/复用 id（把 src_video 当成万能目标），导致"删掉有bus的片段"被解释成
   "删掉整段视频"，或后续操作引用已消失的 id 而整批失败。
2. **解析**：模型偶发返回带解释文字、代码块围栏或多个片段的输出，之前 json.loads
   直接抛 "Extra data"。现改为 _extract_json_array() 容错提取 + 失败自动重试一次。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .timeline import Timeline, Clip
from .footage_index import FootageIndex
from .operations import apply_operations


# ---------------------------------------------------------------------------
# 1) LLM 提供方接口（真实大模型热插拔点）
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: Optional[str] = None) -> str:
        """给定提示词，返回模型原始文本输出（真实后端实现：DeepSeek/OpenAI/本地 Qwen）。"""
        ...


# 中文物体词 → COCO 英文标签（映射表可持续扩充；FootageIndex 的 label 为英文）
_ZH2EN = {
    "公交车": "bus", "巴士": "bus", "车": "car", "汽车": "car", "卡车": "truck",
    "人": "person", "人物": "person", "行人": "person",
    "狗": "dog", "猫": "cat", "自行车": "bicycle", "单车": "bicycle",
    "摩托": "motorcycle", "摩托车": "motorcycle",
}

# 契约 v1.0.0 允许的操作类型（超出即视为模型幻觉，直接报错而不是让它继续污染操作链）
OPS_V1 = ("cut", "trim", "add_subtitle", "add_bgm",
          "delete_clip", "delete_range", "move_clip", "transition", "effect")


def _speech_by_shot(index: FootageIndex) -> List[Tuple[float, float, List[str]]]:
    """把台词按"它覆盖的镜头"聚合为 `(镜头起, 镜头止, [台词…])`。

    为什么事实块里不逐条列台词原区间（如 `8.15-9.34s "第三幕"`）：
    台词段的时间戳是**语音段落边界**（实测 1–3s 一段，可能只是半句话），LLM 若直接
    拿它去 delete_range，会删掉半个镜头、留下剩下半个 —— 与语义删除的
    「区间必须以镜头为单位」是同一条铁律。按镜头聚合后，模型拿到的是
    **可直接使用的编辑区间**，不需要它自己推演外扩（推演极易出错）。

    没有镜头信息时返回空列表，调用方退化为逐条列台词。
    """
    scenes = index.scenes or []
    if not scenes or not index.transcript_segments:
        return []
    agg: Dict[Tuple[float, float], List[str]] = {}
    order: List[Tuple[float, float]] = []
    for t in index.transcript_segments:
        hit = [sc for sc in scenes
               if float(sc["end"]) > t.start + 1e-9 and float(sc["start"]) < t.end - 1e-9]
        if not hit:
            continue
        key = (min(float(sc["start"]) for sc in hit), max(float(sc["end"]) for sc in hit))
        if key not in agg:
            agg[key] = []
            order.append(key)
        agg[key].append(t.text or "")
    return [(a, b, agg[(a, b)]) for (a, b) in order]


def _clip_at(tl: Timeline, t: float) -> Optional[Clip]:
    """返回时间轴时刻 t 落在哪个 clip 内（含端点）。"""
    for c in tl.clips:
        if c.in_ <= t <= c.out:
            return c
    return None


def _label_ranges(index: FootageIndex, label_zh: str) -> List[Tuple[float, float]]:
    """把中文/英文标签在 FootageIndex 中匹配到的视觉段时间区间取出。"""
    label = _ZH2EN.get(label_zh, label_zh)
    return [(s.start, s.end) for s in index.visual_segments if s.label == label]


def _speech_ranges(index: FootageIndex, keyword: str) -> List[Tuple[float, float]]:
    """在**转写段**里找台词包含 keyword 的段，返回它们覆盖的镜头区间。

    台词段的时间戳来自 M2 的 VAD 分段（粒度实测可到"半句"，见 understand._get_vad）。

    **为什么要外扩到镜头**（而不是直接用台词段的区间）：与语义删除同一条铁律 ——
    编辑区间必须以**镜头**为单位。只删台词段那 1–2s，用户看到的会是"这句话说了一半
    画面就跳了"，而同一镜头剩下的部分还在。外扩后得到的是完整的镜头。
    实测素材2：台词段 `[8.15,9.34]`（"第三幕"）→ 扩到幕3 = `(8.0,12.0)`。
    """
    if not keyword:
        return []
    times = [(t.start, t.end) for t in index.transcript_segments
             if keyword in (t.text or "")]
    if not times:
        return []
    scenes = index.scenes or []
    out: List[Tuple[float, float]] = []
    for (s, e) in times:
        hit = [sc for sc in scenes
               if float(sc["end"]) > s + 1e-9 and float(sc["start"]) < e - 1e-9]
        span = (min(float(sc["start"]) for sc in hit),
                max(float(sc["end"]) for sc in hit)) if hit else (s, e)
        if span not in out:                      # 多个台词段常扩到同一镜头，去重
            out.append(span)
    return out


# 「删掉说 X 的片段」的指令形态 —— 规则规划器与降级提示共用同一份正则，
# 避免两处各写一份、日后只改一处而悄悄分家。
_SPEECH_DELETE_RE = re.compile(
    r"(?:删掉|去掉|剪掉|删除)\s*(?:说|讲|提到|念到|说过)"
    r"\s*['\"「]?(.+?)['\"」]?\s*的?\s*(?:片段|镜头|部分|那句|台词)")


def speech_delete_degrades_to_raw_segments(command: str,
                                           index: Optional[FootageIndex]) -> bool:
    """判断这次「按台词删除」是否**降级**成了语音段落区间（而不是完整镜头）。

    降级条件：命中了台词关键词，但 `_speech_by_shot` 聚合不出任何镜头
    （没有镜头信息，或台词与所有镜头都不重叠）。此时 `_speech_ranges` 只能
    退化为台词段自己的 1–2s 区间 —— 正是"只删掉半句、留下半个镜头"的那种结果。

    为什么要把它显式暴露出来：这类降级**不会报错**，成片也"看起来做了点什么"，
    但用户会觉得"没删干净"（Bug N 的同一类症状）。按本项目已确立的原则 ——
    **静默降级等于静默的数据丢失** —— UI 必须在状态里说一句，而不是默默降级。
    """
    if not command or index is None:
        return False
    m = _SPEECH_DELETE_RE.search(command.strip())
    if not m:
        return False
    if not _speech_ranges(index, m.group(1).strip()):
        return False                       # 压根没命中台词，谈不上降级
    return not _speech_by_shot(index)       # 聚合不出镜头 = 只能给语音段落


# 「删掉有 X 的片段」的指令形态 —— 与规则规划器分支 (a) 共用同一份写法。
_SEMANTIC_DELETE_RE = re.compile(
    r"删掉有(.+?)的(片段|镜头|shot)|去掉包含(.+?)的(片段|镜头)?")

# 区间比对容差：模型给出的秒数可能与事实块差一点点（保留 2 位），别因此报虚假差异。
_RANGE_TOL = 0.05


def expected_semantic_ranges(command: str,
                             index: Optional[FootageIndex]
                             ) -> Optional[List[Tuple[float, float]]]:
    """这条指令**究竟该删哪些区间** —— 由 M2 事实确定性算出，与模型无关。

    语义删除（"删掉有 X 的片段" / "删掉说 X 的片段"）本质上不是"创作"，是**检索**：
    答案在 YOLO 标签和 ASR 转写里已经写死了。所以这里可以算出标准答案，
    拿去核对模型的输出 —— 这是 2026-09-24 真机跑 Qwen3-0.6B 暴露出的缺口
    （见 `_verify/verify_llm_real.py`：模型漏删第 3 个公交镜头、又多删了没人要的第 6 幕，
    两条都是**不报错、0 警告**的静默错误）。

    返回 `None` = 这条指令不是语义删除，不该做这项校验；
    返回空 list = 是语义删除但事实里确实没命中（由规划层负责报"未匹配到操作"）。
    """
    if not command or index is None:
        return None
    c = command.strip()
    m = _SEMANTIC_DELETE_RE.search(c)
    if m:                                   # 「删掉有 X 的片段」：视觉优先、台词兜底
        key = (m.group(1) or m.group(3) or "").strip().strip("'\"「」")
        return _label_ranges(index, key) or _speech_ranges(index, key)
    m = _SPEECH_DELETE_RE.search(c)
    if m:                                   # 「删掉说 X 的片段」：走台词检索
        return _speech_ranges(index, m.group(1).strip())
    return None


def semantic_delete_discrepancies(command: str, index: Optional[FootageIndex],
                                  ops: List[Dict[str, Any]]) -> List[str]:
    """拿 M2 事实核对该删除区间，返回可读的差异说明（空 = 一致）。

    **为什么必须有这道闸**：提示词只能**降低**模型犯错率，不能消除。真机实测里
    0.6B 模型在三个公交镜头上只删了 2 个（成片里还剩一段 bus），另一题又多删了
    用户没要的一段 —— **两次都是"操作成功、0 警告"**。这是本项目最危险的一类失效：
    操作被接受、时间轴也真的改了，但结果和用户指令对不上，而没人吱声。

    注意本函数**只报不改**：是"让用户看见"，还是"让系统自动改用确定性结果"，
    交给调用方决定 —— 静默自动纠正会把模型的活悄悄换成规则的活，同样是信息丢失。
    """
    key = _semantic_delete_key(command)
    if key is None:
        return []                            # 不是语义删除，无从核对

    got: List[Tuple[float, float]] = []
    for o in ops or []:
        if not isinstance(o, dict) or o.get("type") != "delete_range":
            continue
        p = o.get("payload") or {}
        s, e = p.get("start"), p.get("end")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)):
            got.append((float(s), float(e)))
    if not got:
        return []                            # 模型没给删除区间，别的环节会报"未匹配"

    # 死角①：手里根本没有 M2 事实。此时模型也只能臆测 —— 报"多删"会误导，
    # 因为真因不是区间算错，而是**没跑理解**。必须把用户指向第 ② 步。
    if not _index_has_m2_facts(index):
        return [f"这条指令要靠素材理解（M2）事实来定位，但当前**没有可用的 M2 结果**"
                f"（未摄取或未跑 ② 理解）。模型给出的 {len(got)} 个删除区间没有事实依据，"
                f"多半是臆测 —— 请先在 ② 跑一次素材理解，再回来规划。"]

    expected = expected_semantic_ranges(command, index) or []
    # 死角②：有事实但一个都没命中。这时说"多删"同样不准确 —— 要说从未命中。
    if not expected:
        return [f"素材里没有任何内容命中「{key}」，模型给出的 {len(got)} 个删除区间"
                f"没有事实依据。请确认关键词是否出现在画面/台词里。"]

    def finds(rng: Tuple[float, float], pool) -> bool:
        return any(abs(a - rng[0]) <= _RANGE_TOL and abs(b - rng[1]) <= _RANGE_TOL
                   for (a, b) in pool)

    msgs: List[str] = []
    missing = [r for r in expected if not finds(r, got)]
    extra = [r for r in got if not finds(r, expected)]
    for (a, b) in missing:
        msgs.append(f"漏删：{a:.2f}-{b:.2f}s 这段也命中了，但模型没给 delete_range"
                    f"（成片里这段内容会保留下来）")
    for (a, b) in extra:
        msgs.append(f"多删：{a:.2f}-{b:.2f}s 不在命中结果里，用户没有要求删除它")
    return msgs


def _semantic_delete_key(command: str) -> Optional[str]:
    """指令是「删掉有/说 X 的片段」时返回关键词；否则 None。"""
    c = (command or "").strip()
    m = _SEMANTIC_DELETE_RE.search(c)
    if m:
        return (m.group(1) or m.group(3) or "").strip().strip("'\"「」") or None
    m = _SPEECH_DELETE_RE.search(c)
    if m:
        return m.group(1).strip() or None
    return None


def _index_has_m2_facts(index: Optional[FootageIndex]) -> bool:
    """有没有可用的 M2 事实（镜头 / 视觉标签 / 台词，任有一条就算有）。"""
    if index is None:
        return False
    return any((index.scenes, index.visual_segments, index.transcript_segments))


# ---------------------------------------------------------------------------
# 2) 规则 Planner（确定性、离线、可单测）
# ---------------------------------------------------------------------------
class RuleBasedPlanner:
    """用正则把常见剪辑指令映射为 M3 操作。覆盖 M4 冒烟所需的代表性子集。"""

    def plan(self, command: str, timeline: Timeline,
             index: Optional[FootageIndex] = None) -> List[Dict[str, Any]]:
        ops: List[Dict[str, Any]] = []
        c = command.strip()

        # (a) 语义检索：删掉有 <label> 的片段（需 FootageIndex）
        #     视觉优先（画面是主要编辑对象）；视觉没匹配到就**回退台词检索** ——
        #     这样"删掉包含『公交车』的片段"在视觉标签是英文 bus 时也能靠台词命中。
        #     正则用模块级 `_SEMANTIC_DELETE_RE`，与 `expected_semantic_ranges` 共用一份写法。
        m = _SEMANTIC_DELETE_RE.search(c)
        if m and index is not None:
            key = (m.group(1) or m.group(3) or "").strip().strip("'\"「」")
            for (s, e) in (_label_ranges(index, key) or _speech_ranges(index, key)):
                ops.extend(self._remove_range(timeline, s, e))

        # (a2) 台词检索：删掉**说/提到** <关键词> 的片段（需 index.transcript_segments）
        #      区间同样外扩到镜头（`_speech_ranges` 内处理）—— 只删那一两秒会留下半个镜头。
        m = _SPEECH_DELETE_RE.search(c)
        if m and index is not None:
            for (s, e) in _speech_ranges(index, m.group(1).strip()):
                ops.extend(self._remove_range(timeline, s, e))

        # (b) 在 T 秒切一刀（允许"在第2秒"里的 第）
        m = re.search(
            r"(?:在|于)\s*第?\s*(\d+(?:\.\d+)?)\s*秒\s*(?:处\s*)?(?:切|分割)"
            r"|切\s*(?:一\s*)?刀?\s*(?:在|于)?\s*第?\s*(\d+(?:\.\d+)?)\s*秒", c)
        if m:
            t = float(m.group(1) or m.group(2))
            clip = _clip_at(timeline, t)
            if clip:
                ops.append({"type": "cut", "payload": {"targetClipId": clip.id, "at": t}})

        # (c) 去掉前 N 秒 / 后 N 秒
        m = re.search(r"去掉前(\d+(?:\.\d+)?)秒|剪掉开头(\d+(?:\.\d+)?)秒|裁掉前(\d+(?:\.\d+)?)秒", c)
        if m:
            n = float(m.group(1) or m.group(2) or m.group(3))
            ops.append({"type": "trim", "payload": {"targetClipId": "src_video", "in": n}})
        m = re.search(r"去掉后(\d+(?:\.\d+)?)秒|剪掉结尾(\d+(?:\.\d+)?)秒|裁掉后(\d+(?:\.\d+)?)秒", c)
        if m:
            n = float(m.group(1))
            ops.append({"type": "trim",
                        "payload": {"targetClipId": "src_video",
                                    "out": timeline.metadata.duration - n}})

        # (d) 加字幕 "文本" [从A秒到B秒]
        m = re.search(
            r"字幕['\"](.+?)['\"]"
            r"(?:\s*(?:从|在)?\s*(\d+(?:\.\d+)?)\s*秒?\s*(?:到|[-~])\s*(\d+(?:\.\d+)?)\s*秒)?", c)
        if m:
            text = m.group(1)
            if m.group(2) and m.group(3):
                st, en = float(m.group(2)), float(m.group(3))
            else:
                st, en = 0.0, min(5.0, timeline.metadata.duration)
            ops.append({"type": "add_subtitle",
                        "payload": {"text": text, "start": st, "end": en}})

        # (e) 删除片段 <id>
        m = re.search(r"删除\s*(?:片段|clip)?\s*[:：]?\s*([\w]+)", c)
        if m:
            ops.append({"type": "delete_clip", "payload": {"targetClipId": m.group(1)}})

        # (f) 画面效果：整体调色/滤镜（作用在 src_video 上）
        for kw, et, params in (
            ("黑白", "grayscale", {}), ("灰度", "grayscale", {}),
            ("复古", "sepia", {}), ("反色", "invert", {}),
            ("水平翻转", "mirror", {}), ("镜像", "mirror", {}),
            ("模糊", "blur", {"radius": 5}),
            ("淡入", "fade_in", {"duration": 1.0}),
            ("淡出", "fade_out", {"duration": 1.0}),
        ):
            if kw in c:
                ops.append({"type": "effect",
                            "payload": {"targetClipId": "src_video",
                                        "effectType": et, "params": params}})
                break
        m = re.search(r"(?:亮度|调亮|变亮)\s*([+-]?\d*\.?\d+)?", c)
        if m and any(k in c for k in ("亮度", "调亮", "变亮")):
            v = float(m.group(1)) if m.group(1) else 0.2
            ops.append({"type": "effect",
                        "payload": {"targetClipId": "src_video",
                                    "effectType": "brightness", "params": {"value": v}}})
        m = re.search(r"(?:对比度)\s*([+-]?\d*\.?\d+)?", c)
        if m:
            v = float(m.group(1)) if m.group(1) else 1.5
            ops.append({"type": "effect",
                        "payload": {"targetClipId": "src_video",
                                    "effectType": "contrast", "params": {"value": v}}})
        m = re.search(r"(?:饱和度)\s*([+-]?\d*\.?\d+)?", c)
        if m:
            v = float(m.group(1)) if m.group(1) else 1.5
            ops.append({"type": "effect",
                        "payload": {"targetClipId": "src_video",
                                    "effectType": "saturation", "params": {"value": v}}})

        return ops

    @staticmethod
    def _remove_range(tl: Timeline, start: float, end: float) -> List[Dict[str, Any]]:
        """删除素材绝对时间轴区间 [start,end]。

        改用 delete_range 单条表达：先前版本手推 `cid__b__a` 这类派生 id，
        多个区间连续操作时后一条会因前一条改动了 id 而失效，被非严格模式静默跳过
        —— 实测「两段 bus 只删掉了后面那段」。delete_range 按素材秒数定位，
        区间之间彼此独立，且执行层会同步删除配对音频（保持 A/V 对齐）。
        """
        return [{"type": "delete_range", "payload": {"start": start, "end": end}}]


# ---------------------------------------------------------------------------
# 2.5) 提示词上下文拼装（LLM 看到的"事实"，必须与真实 Timeline 一致）
# ---------------------------------------------------------------------------
def timeline_digest(tl: Timeline) -> str:
    """Timeline 的紧凑摘要：**可用 clip id 清单**（LLM 只能引用这些 id）。"""
    lines = []
    for c in tl.clips:
        lines.append(f"- `{c.id}` | track={c.track} | {c.in_:.2f}–{c.out:.2f}s "
                     f"| 素材={c.sourceRef}")
    if tl.subtitles:
        lines.append("- 已有字幕：" + "; ".join(
            f"{s.start:.2f}–{s.end:.2f}s “{s.text}”" for s in tl.subtitles))
    lines.append(f"- 时间轴总时长 {tl.metadata.duration:.2f}s")
    return "\n".join(lines)


def facts_block(index: Optional[FootageIndex]) -> str:
    """M2 素材理解事实：镜头 / 检出物体时间区间 / 台词。无 index 时显式声明缺失。"""
    if index is None:
        return "（未摄取素材：无镜头/物体/台词事实，语义类指令无法规划）"
    lines: List[str] = []
    if index.scenes:
        lines.append("镜头区间：" + "; ".join(
            f"{s['start']:.2f}-{s['end']:.2f}s" for s in index.scenes))
    agg: Dict[str, List[Tuple[float, float]]] = {}
    for v in index.visual_segments:
        agg.setdefault(v.label, []).append((v.start, v.end))
    if agg:
        lines.append("检出物体（标签: 时间区间）：" + "; ".join(
            f"{k}: " + ",".join(f"{a:.2f}-{b:.2f}s" for a, b in v) for k, v in agg.items()))
    else:
        lines.append("检出物体：（空 —— 未跑 M2 或未检出；涉及物体/人物的指令无法规划）")
    if index.transcript_segments:
        grouped = _speech_by_shot(index)
        if grouped:
            # 按镜头聚合后给模型"可直接删除的区间" —— 避免它拿半句台词的时间戳去删，
            # 留下半个镜头（详见 `_speech_by_shot` 的说明）。
            lines.append("台词（**按镜头聚合：要按台词删除，请直接用下面的镜头区间**）：" + "; ".join(
                f"镜头 {a:.2f}-{b:.2f}s: " + " / ".join(f"“{t}”" for t in texts)
                for (a, b, texts) in grouped))
        else:
            # 退化分支必须**自报家门**：这里没有「镜头 X-Y」前缀，而 schema 的 3b 规则
            # 默认台词是"已按镜头聚合"的。若不显式说明，提示词就在对模型说假话 ——
            # 模型会去找不存在的「镜头 X-Y」，或者干脆违反 3b 用台词小时间戳去删。
            lines.append("台词（**未按镜头聚合：没有可用的镜头信息，下面的台词区间就是能给的**）："
                         + "; ".join(f"{t.start:.2f}-{t.end:.2f}s “{t.text}”"
                                     for t in index.transcript_segments))
    lines.append(f"素材时长：{float(index.metadata.get('duration', 0.0)):.2f}s")
    return "\n".join(f"- {x}" for x in lines)


# ---------------------------------------------------------------------------
# 3) 真实 LLM Planner（热插拔点）
# ---------------------------------------------------------------------------
class JsonPlanner:
    """真实 LLM 路径：把操作 schema + 真实上下文放进提示词，解析模型输出的 JSON 操作列表。"""

    _SCHEMA_HINT = (
        "你是视频剪辑智能体的规划器。你的唯一输出是一个 JSON 数组（不能有解释文字、"
        "不能有 Markdown 代码块）。每个元素形如 "
        '{"type":"<操作名>","payload":{...}}。\n'
        "\n"
        "【可用操作与字段】\n"
        "- cut: {targetClipId, at}          在 at 秒把该 clip 切成两段\n"
        "- trim: {targetClipId, in?, out?}  只保留 [in,out] 秒区间（去掉前N秒=in:N；"
        "去掉后N秒=out:总时长-N）\n"
        "- delete_clip: {targetClipId}      删除一个 clip（视频段被删时其配对音频会一起删）\n"
        "- delete_range: {start, end}       【删除内容推荐写法】按素材绝对秒数删除 [start,end]，"
        "视频与其配对音频一起删；不依赖派生 id\n"
        "- add_subtitle: {text, start, end}\n"
        "- add_bgm: {sourceRef, in, out, volume?, fadeIn?, fadeOut?}  铺背景音乐，"
        "与主音频混音（不会盖掉原声）\n"
        "- move_clip: {targetClipId, toTrack?, toIn?}  把片段挪到 toIn 秒处，"
        "整条轨会重排为顺序相接\n"
        "- transition: {transitionType, duration, clips:[前一个id, 后一个id]}  "
        "片段交界处淡出/淡入（transitionType 用 fade；fadewhite 为淡入淡出到白）\n"
        "- effect: {targetClipId, effectType, params?}  画面效果 effectType 只能用 "
        "\"grayscale\"/\"blur\"/\"brightness\"/\"contrast\"/\"saturation\"/\"sepia\"/"
        "\"invert\"/\"mirror\"/\"fade_in\"/\"fade_out\""
        "（配 params 如 {\"value\":0.2}、{\"radius\":5}、{\"duration\":1.0}）；"
        "音轨上可用 \"volume\"/\"fade_in\"/\"fade_out\"\n"
        "\n"
        "【铁律：违反会被拒绝执行】\n"
        "1. targetClipId 只能取『当前 Timeline』里列出的 id，绝不能臆造或凭记忆写。\n"
        "2. 操作按数组顺序依次执行。cut 会把该 clip 拆成两个新 clip："
        "`<id>__a`（前半）与 `<id>__b`（后半），**原 id 随即消失**。"
        "所以 cut 之后要继续引用那一段，必须改用新 id。\n"
        "3. 删除某类已检出的内容（如\"删掉有 bus 的片段\"）：对『素材理解事实』里该标签的"
        "**每一个时间区间**各输出一条 delete_range，start/end 直接用区间的绝对秒数。"
        "**绝对不要**自己推演 cut 派生出来的 `__b__b__a` 这类 id —— 多个区间接连操作时"
        "极易失效，导致删除被静默跳过。\n"
        "3b. 按**台词**删除（如\"删掉说公交车的那句\"）：先看事实里台词那行的开头 ——\n"
        "    · 写着\"**按镜头聚合**\"：每条前面的「镜头 X-Y」就是可直接使用的删除区间，"
        "照它各输出一条 delete_range；\n"
        "    · 写着\"**未按镜头聚合**\"（没有镜头信息）：直接用台词自己给的区间。\n"
        "    两种情况都**不要**自己推演或编造镜头边界。"
        "注意台词本身的小时间戳（往往只有一两秒）是语音段落边界，"
        "拿它去删会只删掉半句、留下半个镜头 —— 也正因如此，能用「镜头 X-Y」时就必须用它。\n"
        "4. 除非用户明确要求删掉整段视频/整条音轨，否则不要对 src_video / src_audio 用 delete_clip。\n"
        "5. 指令无法用以上操作表达时，输出 []。\n"
        "\n"
        "【示例】事实里 bus 出现在 0.50–4.00s 与 10.50–14.00s，指令\"删掉有bus的片段\" ->\n"
        '[{"type":"delete_range","payload":{"start":0.5,"end":4.0}},'
        '{"type":"delete_range","payload":{"start":10.5,"end":14.0}}]\n'
        "（每个区间一条，彼此独立、与顺序无关；系统会自动把对应的音频一起删掉。）\n"
        "\n"
        "【示例：按台词删除】事实里写着\n"
        "- 台词（按镜头聚合：要按台词删除，请直接用下面的镜头区间）："
        "镜头 0.00-4.00s: “第一幕” / “画面里有一辆公交车”; "
        "镜头 8.00-12.00s: “第三幕” / “又出现了一辆公交车”\n"
        '指令"删掉说公交车的片段" -> 【注意】要用「镜头 X-Y」，不是台词自己的小时间戳 ->\n'
        '[{"type":"delete_range","payload":{"start":0.0,"end":4.0}},'
        '{"type":"delete_range","payload":{"start":8.0,"end":12.0}}]\n'
        "（看到「镜头」就照抄它的起止秒；台词冒号后面那些文字只用来判断该句里有没有关键词。）\n"
        "\n"
        "【示例：在某一秒切一刀（需要精确 id 时）】Timeline 只有 src_video(0–15s)，"
        '指令"在第2秒切一刀" -> [{"type":"cut","payload":{"targetClipId":"src_video","at":2.0}}]\n'
        "\n"
        "单步示例：\n"
        '指令"去掉前2秒" -> [{"type":"trim","payload":{"targetClipId":"src_video","in":2.0}}]\n'
        '指令\'加字幕"开场"\' -> '
        '[{"type":"add_subtitle","payload":{"text":"开场","start":0.0,"end":5.0}}]'
    )

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def build_prompt(self, command: str, timeline: Timeline,
                     index: Optional[FootageIndex] = None) -> str:
        return (
            f"{self._SCHEMA_HINT}\n\n"
            f"【当前 Timeline（targetClipId 只能取这里列出的 id）】\n{timeline_digest(timeline)}\n\n"
            f"【当前 Timeline 原始 JSON】\n{timeline.dumps()}\n\n"
            f"【素材理解事实（M2）】\n{facts_block(index)}\n\n"
            f"【用户指令】{command}\n\n"
            f"只输出 JSON 数组。"
        )

    def plan(self, command: str, timeline: Timeline,
             index: Optional[FootageIndex] = None) -> List[Dict[str, Any]]:
        prompt = self.build_prompt(command, timeline, index)
        raw = self.provider.complete(prompt)
        try:
            data = _extract_json_array(raw)
        except ValueError as first_err:
            # 模型偶发夹带解释文字/多个片段：追加一次强约束重试（temperature=0 下多为解析问题）
            retry = prompt + ("\n\n注意：上一次输出无法解析。请只输出一个 JSON 数组，"
                              "不要任何解释、前后缀或 Markdown 代码块。")
            raw2 = self.provider.complete(retry)
            try:
                data = _extract_json_array(raw2)
            except ValueError:
                raise ValueError(
                    f"LLM 输出无法解析为 JSON 操作数组；"
                    f"首次输出片段：{(raw or '')[:150]!r}；"
                    f"重试输出片段：{(raw2 or '')[:150]!r}") from first_err
        return _validate_ops(data, raw)


def _extract_json_array(text: str) -> List[Any]:
    """从模型输出中稳健提取 JSON 数组（容忍代码块围栏、前后解释、多个片段）。"""
    t = (text or "").strip()
    if not t:
        raise ValueError("模型输出为空")
    fence = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        data = json.loads(t)
        if isinstance(data, list):
            return data
    except Exception:                               # noqa: BLE001
        pass
    # 扫描第一个括号平衡的数组块（跳过字符串内的括号）
    start = t.find("[")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(t[start:i + 1])
                        if isinstance(data, list):
                            return data
                    except Exception:               # noqa: BLE001
                        pass
                    break
        start = t.find("[", start + 1)
    raise ValueError(f"未找到合法的 JSON 数组：{t[:200]!r}")


def _validate_ops(data: Any, raw: str = "") -> List[Dict[str, Any]]:
    """校验形状：必须是 [{type: 合法操作名, payload: dict}, ...]。"""
    if not isinstance(data, list):
        raise ValueError(f"LLM 输出不是 JSON 数组：{str(data)[:200]!r}")
    ops: List[Dict[str, Any]] = []
    for i, o in enumerate(data, 1):
        if not isinstance(o, dict):
            raise ValueError(f"第 {i} 个元素不是对象：{o!r}")
        typ = o.get("type")
        if typ not in OPS_V1:
            raise ValueError(f"第 {i} 个元素操作类型非法 `{typ}`；"
                             f"契约仅支持 {list(OPS_V1)}。原始输出片段：{raw[:200]!r}")
        o.setdefault("payload", {})
        if not isinstance(o["payload"], dict):
            raise ValueError(f"第 {i} 个元素的 payload 不是对象：{o['payload']!r}")
        ops.append(o)
    return ops


# ---------------------------------------------------------------------------
# 4) 编辑智能体（对外门面）
# ---------------------------------------------------------------------------
class EditingAgent:
    """把自然语言指令规划为结构化操作；可选接入真实 LLM。"""

    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider
        self._rule = RuleBasedPlanner()

    def plan(self, command: str, timeline: Timeline,
             index: Optional[FootageIndex] = None) -> List[Dict[str, Any]]:
        """返回操作列表 [{type, payload}, ...]。有 provider 走 JsonPlanner，否则规则 Planner。"""
        if self.provider is not None:
            return JsonPlanner(self.provider).plan(command, timeline, index)
        return self._rule.plan(command, timeline, index)

    def execute(self, command: str, timeline: Timeline,
                index: Optional[FootageIndex] = None) -> Timeline:
        """便捷：规划并批量 apply，返回新 Timeline（不修改入参）。

        走 apply_operations(strict=True)：任何一条操作失败都抛带上下文的
        OperationError（而不是裸 KeyError），便于定位是哪条指令的问题。
        """
        ops = self.plan(command, timeline, index)
        new_tl, _logs, _warns = apply_operations(timeline, ops, strict=True)
        return new_tl
