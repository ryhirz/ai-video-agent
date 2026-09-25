"""确定性编译器：Timeline -> FFmpeg 命令行（纯函数，可单测/可回放）。

对应规格书 §3。compile(timeline) 只读 timeline，产出 filter_complex 命令。
同输入必得同输出；节点命名带 clipId，出错可反查具体操作。

覆盖范围（2026-09-23 补齐）：契约里的 9 类操作**全部**会体现在成片里——
    cut / trim / delete_clip / delete_range  -> 影响参与拼接的片段区间
    move_clip                                -> 影响片段顺序（重排语义）
    add_subtitle                             -> ASS 叠加
    effect                                   -> 逐片段画面/音频滤镜
    transition                               -> 片段边界的淡出/淡入
    add_bgm                                  -> 与主音频 amix 混音
**未知效果不会被静默忽略**：识别不了的写进返回值的 `unsupported`，由上层回显给用户。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .timeline import Timeline, Clip

POS_MAP = {
    "bottom-center": "\\an2", "bottom-left": "\\an1", "bottom-right": "\\an3",
    "top-center": "\\an8", "top-left": "\\an7", "top-right": "\\an9", "center": "\\an5",
}

# ---------------------------------------------------------------------------
# 效果词表
# ---------------------------------------------------------------------------
# 画面效果：固定滤镜
_SIMPLE_VIDEO_EFFECTS = {
    "grayscale": "hue=s=0", "灰度": "hue=s=0", "黑白": "hue=s=0",
    "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    "复古": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    "invert": "negate", "反色": "negate",
    "mirror": "hflip", "hflip": "hflip", "水平翻转": "hflip",
    "vflip": "vflip", "垂直翻转": "vflip",
}
# 画面效果：需要参数（value / radius）
_PARAM_VIDEO_EFFECTS = {
    "blur": ("boxblur", "radius", 5, "{v}"),
    "模糊": ("boxblur", "radius", 5, "{v}"),
    "brightness": ("eq", "value", 0.1, "brightness={v}"),
    "亮度": ("eq", "value", 0.1, "brightness={v}"),
    "contrast": ("eq", "value", 1.5, "contrast={v}"),
    "对比度": ("eq", "value", 1.5, "contrast={v}"),
    "saturation": ("eq", "value", 1.5, "saturation={v}"),
    "饱和度": ("eq", "value", 1.5, "saturation={v}"),
}
VIDEO_EFFECTS: Tuple[str, ...] = tuple(
    sorted(set(_SIMPLE_VIDEO_EFFECTS) | set(_PARAM_VIDEO_EFFECTS)
           | {"fade_in", "淡入", "fade_out", "淡出"}))

# 音频效果（作用在音轨 clip 上）
AUDIO_EFFECTS: Tuple[str, ...] = (
    "volume", "音量", "fade_in", "淡入", "fade_out", "淡出")

# add_bgm 在 clip 上打的内部标记（不是画面效果，编译器单独处理）
BGM_MARKER = "bgm_mix"


def _track_type(tl: Timeline, c: Clip) -> str:
    for tr in tl.tracks:
        if tr.id == c.track:
            return tr.type
    return "video"


def _ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _num(params: Dict[str, Any], names: Tuple[str, ...], default: float) -> float:
    for n in names:
        v = params.get(n)
        if isinstance(v, (int, float)):
            return float(v)
    return default


def _is_bgm(clip: Clip) -> bool:
    """判断一个 clip 是否为 add_bgm 铺的背景音乐（靠 effect 标记识别）。"""
    return any((e or {}).get("effectType") == BGM_MARKER for e in (clip.effects or []))


def _bgm_params(clip: Clip) -> Dict[str, Any]:
    for e in clip.effects or []:
        if (e or {}).get("effectType") == BGM_MARKER:
            return e.get("params") or {}
    return {}


def video_effect_filter(effect: Dict[str, Any],
                        clip_dur: Optional[float] = None) -> Optional[str]:
    """单个画面效果 -> ffmpeg 滤镜串；不认识返回 None（由调用方记入 unsupported）。

    clip_dur 是片段时长，只有淡入/淡出需要它（要算出淡出的起始时刻）。
    """
    t = str((effect or {}).get("effectType") or "").strip()
    p = (effect or {}).get("params") or {}
    if t in _SIMPLE_VIDEO_EFFECTS:
        return _SIMPLE_VIDEO_EFFECTS[t]
    if t in _PARAM_VIDEO_EFFECTS:
        _filt, pname, default, tmpl = _PARAM_VIDEO_EFFECTS[t]
        v = _num(p, (pname, "value"), default)
        return (f"boxblur={v}" if _filt == "boxblur" else tmpl.format(v=v))
    cd = clip_dur if (clip_dur and clip_dur > 0) else 2.0
    if t in ("fade_in", "淡入"):
        d = _num(p, ("duration", "value"), min(1.0, cd / 2))
        return f"fade=t=in:st=0:d={d:.3f}" if d > 0 else None
    if t in ("fade_out", "淡出"):
        d = _num(p, ("duration", "value"), min(1.0, cd / 2))
        return f"fade=t=out:st={cd - d:.3f}:d={d:.3f}" if 0 < d < cd else None
    return None


def audio_effect_filter(effect: Dict[str, Any], clip_dur: float) -> Optional[str]:
    """单个音频效果 -> ffmpeg 滤镜串；不认识返回 None。"""
    t = str((effect or {}).get("effectType") or "").strip()
    p = (effect or {}).get("params") or {}
    if t in ("volume", "音量"):
        return f"volume={_num(p, ('value', 'volume'), 1.0)}"
    if t in ("fade_in", "淡入"):
        d = _num(p, ("duration", "value"), min(1.0, clip_dur / 2))
        return f"afade=t=in:st=0:d={d:.3f}" if d > 0 else None
    if t in ("fade_out", "淡出"):
        d = _num(p, ("duration", "value"), min(1.0, clip_dur / 2))
        return (f"afade=t=out:st={clip_dur - d:.3f}:d={d:.3f}"
                if 0 < d < clip_dur else None)
    return None


def build_ass_content(tl: Timeline) -> str:
    """字幕 -> ASS（纯函数，确定性）。position 映射到 \\an 对齐。

    尺寸一律按 timeline 分辨率算，**不能写死**：

    - 必须显式给出 `PlayResX/PlayResY`。ASS 不写这两项时 libass 按 **384×288** 解释脚本坐标，
      字号于是变成"只跟画面**高度**成比例"——横屏 1280×720 碰巧装得下，
      竖屏 720×1280 会**横向溢出被裁掉**（实测 6 个汉字撑满 720px，两端各切掉约一个字）。
    - `WrapStyle: 0` 允许自动折行。原值 `2` 表示"只按 \\N 断行"，长字幕不折行、直接溢出画面。
    - 字号取 `min(H*4.8%, W*5.5%)`：同时受宽高两维约束，横竖屏都不会溢出；
      描边/阴影/边距按同比例缩放，换分辨率后观感一致。
    """
    w = int(tl.metadata.resolution.get("w", 1920) or 1920)
    h = int(tl.metadata.resolution.get("h", 1080) or 1080)
    font = max(12, round(min(h * 0.048, w * 0.055)))
    outline = max(1, round(font * 0.06))
    shadow = max(0, round(font * 0.04))
    ml = mr = max(8, round(w * 0.05))
    mv = max(8, round(h * 0.04))

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {w}",
        f"PlayResY: {h}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Noto Sans CJK SC,{font},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H00000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,{ml},{mr},{mv},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for s in tl.subtitles:
        style = s.style or {}
        an = POS_MAP.get(style.get("position", "bottom-center"), "\\an2")
        text = s.text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        # 覆盖标签必须用 {} 包裹，否则 \an2 会被当成普通文字显示出来
        lines.append(f"Dialogue: 0,{_ass_time(s.start)},{_ass_time(s.end)},Default,,0,0,0,,{{{an}}}{text}")
    return "\n".join(lines)


def compile(tl: Timeline) -> Dict[str, Any]:
    """纯函数：只读 tl -> FFmpeg 描述。同输入同输出。"""
    W = tl.metadata.resolution.get("w", 1920)
    H = tl.metadata.resolution.get("h", 1080)
    vclips = sorted([c for c in tl.clips if _track_type(tl, c) == "video"], key=lambda c: c.in_)
    aclips = sorted([c for c in tl.clips if _track_type(tl, c) == "audio"], key=lambda c: c.in_)
    unsupported: List[str] = []

    bgm_ids = {id(c) for c in aclips if _is_bgm(c)}
    main_aclips = [c for c in aclips if id(c) not in bgm_ids]
    bgm_clips = [c for c in aclips if id(c) in bgm_ids]

    # 转场：把 attachedClips 前一个当「淡出」、后一个当「淡入」
    live_ids = {c.id for c in tl.clips}
    trans_out: Dict[str, Tuple[str, float]] = {}
    trans_in: Dict[str, Tuple[str, float]] = {}
    for tr in tl.transitions:
        ids = [i for i in (tr.attachedClips or []) if i in live_ids]
        if len(ids) < 2:
            if tr.attachedClips:
                unsupported.append(
                    f"转场 `{tr.id}` 引用的片段已不存在，已忽略")
            continue
        trans_out[ids[0]] = (tr.type, float(tr.duration))
        trans_in[ids[1]] = (tr.type, float(tr.duration))

    inputs: List[str] = []
    vid_parts: List[str] = []       # 视频逐片段链
    a_parts: List[str] = []         # 主音频逐片段链
    bgm_parts: List[str] = []       # BGM 逐片段链
    vf_tail: List[str] = []         # 视频汇总（concat / 字幕）
    af_tail: List[str] = []         # 音频汇总（concat / amix）
    vlabels: List[str] = []
    alabels: List[str] = []
    bgm_labels: List[str] = []

    # ---- 视频逐片段链 ----
    for i, c in enumerate(vclips):
        inputs.append(c.sourceRef)
        lbl = f"cv{i}"
        dur = c.sourceOut - c.sourceIn
        chain = [f"trim=start={c.sourceIn}:end={c.sourceOut}",
                 "setpts=PTS-STARTPTS", f"scale={W}:{H}"]
        for e in c.effects or []:
            if (e or {}).get("effectType") == BGM_MARKER:
                continue
            f = video_effect_filter(e, dur)
            if f:
                chain.append(f)
            else:
                unsupported.append(
                    f"画面效果 `{(e or {}).get('effectType')}` 不支持，已忽略"
                    f"（可用：{list(VIDEO_EFFECTS)}）")
        if c.id in trans_out:
            kind, td = trans_out[c.id]
            if td > 0 and dur > td + 1e-6:
                chain.append(f"fade=t=out:st={dur - td:.3f}:d={td:.3f}"
                             + (":color=white" if "white" in kind.lower() else ""))
            else:
                unsupported.append(
                    f"转场时长 {td}s 相对片段 `{c.id}`（{dur:.2f}s）过大，已忽略")
        if c.id in trans_in:
            kind, td = trans_in[c.id]
            if td > 0 and dur > td + 1e-6:
                chain.append(f"fade=t=in:st=0:d={td:.3f}"
                             + (":color=white" if "white" in kind.lower() else ""))
        vid_parts.append(f"[{i}:v]" + ",".join(chain) + f"[{lbl}]")
        vlabels.append(f"[{lbl}]")

    # ---- 主音频逐片段链 ----
    for j, c in enumerate(main_aclips):
        idx = len(vclips) + j
        inputs.append(c.sourceRef)
        lbl = f"ca{j}"
        dur = c.sourceOut - c.sourceIn
        chain = [f"atrim=start={c.sourceIn}:end={c.sourceOut}",
                 "asetpts=PTS-STARTPTS"]
        for e in c.effects or []:
            f = audio_effect_filter(e, dur)
            if f:
                chain.append(f)
            else:
                unsupported.append(
                    f"音频效果 `{(e or {}).get('effectType')}` 不支持，已忽略"
                    f"（可用：{list(AUDIO_EFFECTS)}）")
        a_parts.append(f"[{idx}:a]" + ",".join(chain) + f"[{lbl}]")
        alabels.append(f"[{lbl}]")

    # ---- BGM 逐片段链（音量/淡入淡出/延时定位）----
    for k, c in enumerate(bgm_clips):
        idx = len(vclips) + len(main_aclips) + k
        inputs.append(c.sourceRef)
        lbl = f"cb{k}"
        dur = c.sourceOut - c.sourceIn
        prm = _bgm_params(c)
        vol = _num(prm, ("volume",), 1.0)
        fi = _num(prm, ("fadeIn", "fade_in"), 0.0)
        fo = _num(prm, ("fadeOut", "fade_out"), 0.0)
        chain = [f"atrim=start={c.sourceIn}:end={c.sourceOut}",
                 "asetpts=PTS-STARTPTS", f"volume={vol}"]
        if 0 < fi < dur:
            chain.append(f"afade=t=in:st=0:d={fi:.3f}")
        if 0 < fo < dur:
            chain.append(f"afade=t=out:st={dur - fo:.3f}:d={fo:.3f}")
        if c.in_ > 1e-6:                      # 从第 in_ 秒才开始铺 BGM
            chain.append(f"adelay={int(round(c.in_ * 1000))}:all=1")
        bgm_parts.append(f"[{idx}:a]" + ",".join(chain) + f"[{lbl}]")
        bgm_labels.append(f"[{lbl}]")

    # ---- 视频拼接 ----
    vmap = ""
    if len(vclips) == 1:
        vf_tail.append(f"{vlabels[0]}null[vout]")
        vmap = "[vout]"
    elif len(vclips) > 1:
        vf_tail.append("".join(vlabels) + f"concat=n={len(vclips)}:v=1:a=0[vout]")
        vmap = "[vout]"

    # ---- 字幕叠加 ----
    ass = build_ass_content(tl) if tl.subtitles else ""
    if ass and vmap:
        vf_tail.append(f"{vmap}subtitles=subs.ass[vfinal]")
        vmap = "[vfinal]"

    # ---- BGM 汇总：多段 BGM 之间并行混（各段自带 adelay，天然支持错开/重叠）----
    bgm = ""
    if len(bgm_clips) == 1:
        bgm = bgm_labels[0]
    elif len(bgm_clips) > 1:
        af_tail.append("".join(bgm_labels)
                       + f"amix=inputs={len(bgm_clips)}:duration=longest:normalize=0[abgm]")
        bgm = "[abgm]"

    # ---- 主音频拼接 + 与 BGM 混音 ----
    # 无 BGM 时保持历史行为：主音频 concat 直接落 [aout]（对外契约不变）。
    final_label = "[amain]" if bgm else "[aout]"
    amain = ""
    if len(main_aclips) == 1:
        amain = alabels[0]
    elif len(main_aclips) > 1:
        # 顺序拼接必须用 concat；amix 是并行混音（总长只取最长段），与顺次拼接不符。
        af_tail.append("".join(alabels)
                       + f"concat=n={len(main_aclips)}:v=0:a=1{final_label}")
        amain = final_label

    amap = ""
    if amain and bgm:
        # duration=first：成片长度跟主音频（=视频）走；normalize=0 保证主音不被压低。
        af_tail.append(f"{amain}{bgm}amix=inputs=2:duration=first:"
                       f"dropout_transition=0:normalize=0[aout]")
        amap = "[aout]"
    else:
        amap = amain or bgm

    # 多视频轨会被合并成单条输出流（本 MVP 只有一路画面），如实告知而不是假装支持。
    used_video_tracks = {c.track for c in vclips}
    if len(used_video_tracks) > 1:
        unsupported.append(
            f"时间轴上有多个视频轨（{sorted(used_video_tracks)}）会按入点顺序合并成单条输出流，"
            f"跨轨位置差异不会体现在成片里")

    vf = ";".join(vid_parts + vf_tail)
    af = ";".join(a_parts + bgm_parts + af_tail)
    # filter_complex 内的整体顺序：逐片段链 -> 汇总链（依赖前面的标签）
    fc = ";".join(vid_parts + a_parts + bgm_parts + vf_tail + af_tail)

    cmd: List[str] = ["ffmpeg"]
    for inp in inputs:
        cmd += ["-i", inp]
    if fc:
        cmd += ["-filter_complex", fc]
    if vmap:
        cmd += ["-map", vmap]
    if amap:
        cmd += ["-map", amap]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-shortest", "-y", "output.mp4"]

    return {
        "command": " ".join(cmd),
        # 结构化参数列表：供 render 直接 subprocess 执行，避免对命令字符串做空格切分
        # （路径含空格会切错）。保持与 command 完全一致的语义。
        "argv": cmd,
        "inputs": inputs,
        "video_filter": vf,
        "audio_filter": af,
        "ass": ass,
        # 识别不了 / 表达不了的东西在这里如实列出，由上层回显，绝不静默吞掉
        "unsupported": unsupported,
    }
