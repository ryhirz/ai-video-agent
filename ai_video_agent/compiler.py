"""确定性编译器：Timeline -> FFmpeg 命令行（纯函数，可单测/可回放）。

对应规格书 §3。compile(timeline) 只读 timeline，产出 filter_complex 命令。
同输入必得同输出；节点命名带 clipId，出错可反查具体操作。

覆盖范围（2026-10-22 补齐）：契约里的 11 类操作**全部**会体现在成片里——
    cut / trim / delete_clip / delete_range  -> 影响参与拼接的片段区间
    move_clip                                -> 影响片段顺序（重排语义）
    add_subtitle                             -> ASS 叠加（支持 font/size/color/bg 覆盖）
    effect                                   -> 逐片段画面/音频滤镜
    transition                               -> 片段边界 xfade / acrossfade 交叉溶解
    add_bgm                                  -> 与主音频 amix 混音
    speed                                    -> 视频 setpts + 音频 atempo 同步变速
    add_text                                 -> 文字水印/标题（drawtext，九宫格 + margin）
非操作类补充：
    transform（x/y/rotation/opacity/crop）   -> pad / rotate / colorchannelmixer / crop
    组间 gap                                  -> 视频 tpad 黑帧 + 音频 anullsrc 等长静音
**未知效果不会被静默忽略**：识别不了的写进返回值的 `unsupported`，由上层回显给用户。
"""
from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .timeline import Timeline, Clip, TextOverlay, Transition
from .registry import (  # 单一真相：编译 / 校验 / 能力清单共用 registry 的表
    AUDIO_EFFECTS,
    BGM_MARKER,
    VIDEO_EFFECTS,
    PARAM_VIDEO_EFFECTS as _PARAM_VIDEO_EFFECTS,
    SIMPLE_VIDEO_EFFECTS as _SIMPLE_VIDEO_EFFECTS,
    SPEED_MARKER,
    resolve_transition,
    validate_transform,
)

POS_MAP = {
    "bottom-center": "\\an2", "bottom-left": "\\an1", "bottom-right": "\\an3",
    "top-center": "\\an8", "top-left": "\\an7", "top-right": "\\an9", "center": "\\an5",
}

# 效果词表已收敛到 registry.py（原 _SIMPLE_VIDEO_EFFECTS / _PARAM_VIDEO_EFFECTS /
# VIDEO_EFFECTS / AUDIO_EFFECTS 的本地定义已删除），此处仅导入，行为完全不变。


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


def _ass_color(value: Any) -> Optional[str]:
    """#RRGGBB 或 rgb()/rgba() -> ASS 的 AABBGGRR（不含 &H 前缀）。

    ASS 的 alpha 与 CSS 相反：00 = 不透明，FF = 全透明。
    """
    v = str(value or "").strip()
    if not v:
        return None
    if v.startswith("#") and len(v) == 7:
        r, g, b = v[1:3], v[3:5], v[5:7]
        return f"00{b}{g}{r}".upper()
    m = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)", v)
    if m:
        r, g, b = (int(m.group(i)) for i in (1, 2, 3))
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        alpha = int(round((1.0 - max(0.0, min(1.0, a))) * 255))
        return f"{alpha:02X}{b:02X}{g:02X}{r:02X}"
    return None


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


# ---------------------------------------------------------------------------
# transform -> 几何 / 透明度 filter（T4）
# ---------------------------------------------------------------------------
def _fmt_num(v: float) -> str:
    """数字转 filtergraph 字面量：整数不带小数点，浮点去掉多余 0。"""
    return f"{float(v):g}"


def _crop_expr(v: float, axis: str) -> str:
    """crop 参数：0 < v <= 1 视为占比（iw* / ih*），否则按像素。

    边界说明：v == 1.0 按"100% 占比"处理（等价不裁），而不是 1 像素 ——
    "裁 1 像素宽"没有实际意义，因此这个约定是安全的。
    """
    if 0 < v <= 1:
        return f"i{axis}*{_fmt_num(v)}"
    return _fmt_num(v)


def crop_filter(crop: Optional[Dict[str, float]]) -> Optional[str]:
    """transform.crop -> crop 滤镜串（源坐标系，放在 scale 之前）。

    - `w`/`h` 必填；`0 < v <= 1` 视为占比（`iw*0.8`），否则按源像素。
    - `x`/`y` **可以只传一个**：缺的那个用 ffmpeg 默认居中表达式
      （`(iw-ow)/2` / `(ih-oh)/2`）。注意这与 drawtext 的策略不同 ——
      drawtext 的 x/y 是"覆盖九宫格锚点"，必须成对才有意义。
    """
    if not crop:
        return None
    w, h = crop.get("w"), crop.get("h")
    if w is None or h is None:
        return None
    parts = [_crop_expr(float(w), "w"), _crop_expr(float(h), "h")]
    x, y = crop.get("x"), crop.get("y")
    if x is not None or y is not None:
        x_expr = _crop_expr(float(x), "w") if x is not None else "(iw-ow)/2"
        y_expr = _crop_expr(float(y), "h") if y is not None else "(ih-oh)/2"
        parts += [x_expr, y_expr]
    return "crop=" + ":".join(parts)


def transform_geometry_filters(clip: Clip, out_w: int, out_h: int) -> List[str]:
    """clip.transform 的几何部分 -> pad / rotate（+ 尺寸回调）。

    顺序：pad -> rotate -> [scale 回 out_w x out_h]

    - rotation 用 rotw/roth 扩画布后**必须**再 scale 回目标分辨率：否则该片段
      尺寸与其他片段不同，下游 concat 会因尺寸不一致而失败。
    - 含 t 的动态 rotation 表达式无法静态求 rotw/roth，保持原尺寸（ow=iw:oh=ih）。
    - 参数非法时抛 ValueError（编译阶段报错，不交给 ffmpeg）。
    """
    tf = clip.transform
    errs = validate_transform({
        "x": tf.x, "y": tf.y, "scale": tf.scale,
        "rotation": tf.rotation, "opacity": tf.opacity,
        "crop": tf.crop,
    })
    if errs:
        raise ValueError("transform 非法：" + "；".join(errs))

    filters: List[str] = []
    x, y = float(tf.x), float(tf.y)
    if x != 0 or y != 0:
        # 兜底：registry 的 min=0 已在 validate_transform 阶段拦下负值；
        # 这里再挡一次，避免绕过校验直接调用时把非法值交给 ffmpeg。
        if x < 0 or y < 0:
            raise ValueError(
                f"transform.x/y 不支持负值（ffmpeg pad 限制）：x={x:g}, y={y:g}")
        filters.append(f"pad=iw:ih:{_fmt_num(x)}:{_fmt_num(y)}")

    rot = tf.rotation
    if rot != 0:
        if isinstance(rot, str):
            filters.append(f"rotate=a={rot.strip()}:ow=iw:oh=ih:c=black")
        else:
            a = _fmt_num(rot)
            filters.append(f"rotate=a={a}:ow=rotw({a}):oh=roth({a}):c=black")
            filters.append(f"scale={out_w}:{out_h}")
    return filters


def transform_opacity_filters(clip: Clip) -> List[str]:
    """clip.transform.opacity -> format=rgba + colorchannelmixer（仅 <1 时）。"""
    op = float(clip.transform.opacity)
    if op >= 1.0:
        return []
    return ["format=rgba", f"colorchannelmixer=aa={_fmt_num(op)}"]


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


def _speed_rate(clip: Clip) -> Optional[float]:
    """读取片段上的变速倍率（存在 effects 里的 speed 标记）；未设置返回 None。"""
    for e in clip.effects or []:
        if (e or {}).get("effectType") == SPEED_MARKER:
            try:
                return float((e.get("params") or {}).get("rate", 1.0))
            except (TypeError, ValueError):
                return None
    return None


def _atempo_filters(rate: float) -> List[str]:
    """把倍率拆成 atempo 链（单次支持 0.5~100）。rate == 1 返回空列表。

    极端倍率串联：200x -> atempo=100,atempo=2；0.25x -> atempo=0.5,atempo=0.5。
    """
    if abs(float(rate) - 1.0) < 1e-9:
        return []
    filters: List[str] = []
    r = float(rate)
    while r > 100.0:
        filters.append("atempo=100")
        r /= 100.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    if abs(r - 1.0) > 1e-9:
        filters.append(f"atempo={_fmt_num(r)}")
    return filters


# ---------------------------------------------------------------------------
# T7：文字水印 -> drawtext
# ---------------------------------------------------------------------------
# 字体名 -> 各平台候选路径。fontconfig 在 Windows 实测加载失败（无法读取默认配置），
# 所以只能走 fontfile；候选路径按平台挑第一个存在的。
_FONT_CANDIDATES: Dict[str, Dict[str, List[str]]] = {
    "arial": {
        "win32": ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"],
        "linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"],
        "darwin": ["/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Supplemental/Arial.ttf"],
    },
    "microsoft yahei": {
        "win32": ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"],
        "linux": ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"],
        "darwin": ["/System/Library/Fonts/PingFang.ttc"],
    },
    "simhei": {
        "win32": ["C:/Windows/Fonts/simhei.ttf"],
        "linux": ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"],
        "darwin": ["/System/Library/Fonts/PingFang.ttc"],
    },
    "simsun": {
        "win32": ["C:/Windows/Fonts/simsun.ttc"],
        "linux": ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"],
        "darwin": ["/System/Library/Fonts/PingFang.ttc"],
    },
}
# 中文别名
_FONT_CANDIDATES["微软雅黑"] = _FONT_CANDIDATES["microsoft yahei"]
_FONT_CANDIDATES["黑体"] = _FONT_CANDIDATES["simhei"]
_FONT_CANDIDATES["宋体"] = _FONT_CANDIDATES["simsun"]

_PLATFORM_KEY = {"Windows": "win32", "Linux": "linux", "Darwin": "darwin"}


def _resolve_font(name: str) -> Optional[str]:
    """字体名 / 文件路径 -> 可用的字体文件路径；找不到返回 None。"""
    if not name:
        return None
    p = Path(name)
    if p.suffix.lower() in (".ttf", ".ttc", ".otf") and p.exists():
        return str(p)
    cands = _FONT_CANDIDATES.get(name.strip().lower())
    if not cands:
        return None
    key = _PLATFORM_KEY.get(platform.system(), "linux")
    for c in cands.get(key, []):
        if Path(c).exists():
            return c
    return None


def _escape_drawtext(s: str) -> str:
    """转义 drawtext 参数里的 filtergraph 特殊字符（顺序：先 \\，再 '，最后 :）。"""
    return (s.replace("\\", "\\\\")
             .replace("'", "\\'")
             .replace(":", "\\:"))


_POSITION_EXPR: Dict[str, Tuple[str, str]] = {
    "top-left": ("{m}", "{m}"),
    "top-center": ("(w-text_w)/2", "{m}"),
    "top-right": ("w-text_w-{m}", "{m}"),
    "center-left": ("{m}", "(h-text_h)/2"),
    "center": ("(w-text_w)/2", "(h-text_h)/2"),
    "center-right": ("w-text_w-{m}", "(h-text_h)/2"),
    "bottom-left": ("{m}", "h-text_h-{m}"),
    "bottom-center": ("(w-text_w)/2", "h-text_h-{m}"),
    "bottom-right": ("w-text_w-{m}", "h-text_h-{m}"),
}


def drawtext_filter(t: TextOverlay,
                    unsupported: Optional[List[str]] = None) -> str:
    """TextOverlay -> drawtext 滤镜串。

    `x`/`y` 必须**同时非空**才覆盖九宫格锚点：只给一个会让另一半仍是锚点表达式，
    语义混乱。这一点与 crop 的"x/y 可单边"不同，是两套 API 各自的约束。
    """
    m = _fmt_num(t.margin)
    if t.x is not None and t.y is not None:
        x_expr, y_expr = _fmt_num(t.x), _fmt_num(t.y)
    else:
        xe, ye = _POSITION_EXPR.get(t.position, _POSITION_EXPR["bottom-center"])
        x_expr, y_expr = xe.format(m=m), ye.format(m=m)

    parts: List[str] = []
    font_path = _resolve_font(t.font)
    if font_path:
        parts.append(f"fontfile='{_escape_drawtext(font_path)}'")
    elif t.font and unsupported is not None:
        unsupported.append(
            f"字体 `{t.font}` 未找到，已回退默认字体（文字仍会渲染）")

    parts += [
        f"text='{_escape_drawtext(t.text)}'",
        f"fontsize={int(t.size)}",
        f"fontcolor={t.color}",
        f"x={x_expr}",
        f"y={y_expr}",
    ]
    if t.bg:
        parts += ["box=1", f"boxcolor={t.bg}", "boxborderw=6"]
    parts.append(f"enable='between(t,{_fmt_num(t.start)},{_fmt_num(t.end)})'")
    return "drawtext=" + ":".join(parts)


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
        tags = [POS_MAP.get(style.get("position", "bottom-center"), "\\an2")]
        # T6：样式覆盖（font / size / color / bg）。缺省时保持 ASS 全局样式不变。
        if style.get("font"):
            tags.append(f"\\fn{style['font']}")
        if style.get("size"):
            tags.append(f"\\fs{int(float(style['size']))}")
        fg = _ass_color(style.get("color"))
        if fg:
            tags.append(f"\\c&H{fg}&")
        bg = _ass_color(style.get("bg"))
        if bg:
            tags.append(f"\\4c&H{bg}&")
            tags.append("\\bord4")     # BorderStyle=1 下用边框近似底色块
        text = s.text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        # 覆盖标签必须用 {} 包裹，否则 \an2 会被当成普通文字显示出来
        lines.append(
            f"Dialogue: 0,{_ass_time(s.start)},{_ass_time(s.end)},Default,,0,0,0,,"
            f"{{{''.join(tags)}}}{text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T5 真转场：xfade / acrossfade + 纯函数分组
# ---------------------------------------------------------------------------
def _stream_dur(clip: Clip) -> float:
    """片段在流上的实际时长（trim + setpts 之后）。"""
    return float(clip.sourceOut - clip.sourceIn)


def _transition_between(transitions: List[Transition], id_a: str,
                        id_b: str) -> Optional[Transition]:
    """取连接 id_a 与 id_b 的转场（不区分 attachedClips 顺序）。"""
    for tr in transitions:
        ids = list(tr.attachedClips or [])
        if len(ids) == 2 and {ids[0], ids[1]} == {id_a, id_b}:
            return tr
    return None


def _xfade_kind(transition_type: str) -> str:
    """契约 transitionType -> ffmpeg xfade.transition（从 registry 取，单一真相）。"""
    spec = resolve_transition(transition_type)
    return spec.video if spec else "fade"


def xfade_offset(clip_a: Clip, duration: float) -> float:
    """第一对转场的 offset：clipA 流时长 − 转场时长。

    零时长素材、或转场时长不落在 (0, dur) 内时抛 ValueError（契约级约束）。
    """
    dur = _stream_dur(clip_a)
    if dur <= 0:
        raise ValueError(
            f"片段 `{clip_a.id}` 流时长为 0（sourceIn == sourceOut），无法做转场")
    d = float(duration)
    if d <= 0 or d >= dur:
        raise ValueError(
            f"转场时长 {d:g}s 必须落在 (0, {dur:g}s) 内（片段 `{clip_a.id}` 的流时长）")
    return dur - d


def xfade_total_duration(clips: List[Clip],
                         transitions: List[Transition]) -> float:
    """转场后的成片总时长 = Σ片段流时长 − Σ有效转场时长。"""
    total = sum(_stream_dur(c) for c in clips)
    ids = {c.id for c in clips}
    for tr in transitions:
        pair = list(tr.attachedClips or [])
        if len(pair) == 2 and all(i in ids for i in pair):
            total -= float(tr.duration)
    return total


def _valid_transitions(tl: Timeline,
                       unsupported: Optional[List[str]] = None) -> List[Transition]:
    """过滤出「引用完整 + 时长合法」的转场；非法项按需记入 unsupported。"""
    live = {c.id: c for c in tl.clips}

    def _bad(msg: str) -> None:
        if unsupported is not None:
            unsupported.append(msg)

    out: List[Transition] = []
    for tr in tl.transitions:
        ids = list(tr.attachedClips or [])
        if len(ids) != 2 or any(i not in live for i in ids):
            _bad(f"转场 `{tr.id}` 引用的片段不存在或数量不为 2，已忽略")
            continue
        dur_a = _stream_dur(live[ids[0]])
        dur_b = _stream_dur(live[ids[1]])
        d = float(tr.duration)
        shortest = min(dur_a, dur_b)
        if dur_a <= 0 or dur_b <= 0 or d <= 0 or d >= shortest:
            _bad(f"转场 `{tr.id}` 时长 {d:g}s 不合法"
                 f"（两段流时长 {dur_a:g}s / {dur_b:g}s），已忽略并退化为硬切")
            continue
        out.append(tr)
    return out


def compiled_duration(tl: Timeline) -> float:
    """成片时长 = Σ片段流时长 − Σ有效转场时长 + Σ组间 gap（方案 A）。"""
    vclips = sorted([c for c in tl.clips if _track_type(tl, c) == "video"],
                    key=lambda c: c.in_)
    if not vclips:
        return 0.0
    trans = _valid_transitions(tl)
    total = sum(_stream_dur(c) for c in vclips) - sum(float(t.duration) for t in trans)
    groups = group_clips_by_transition(vclips, trans)
    for g1, g2 in zip(groups, groups[1:]):
        gap = g2[0].in_ - g1[-1].out
        if gap > 1e-6:
            total += gap
    return total


def group_clips_by_transition(clips: List[Clip],
                              transitions: List[Transition]) -> List[List[Clip]]:
    """按「相邻对是否有转场」把片段分组（纯函数）。

    不变量（破坏即抛，属内部错误）：
    - 组内相邻对必须有 transition；
    - 组间相邻对必须没有 transition。
    """
    if not clips:
        return []
    ordered = sorted(clips, key=lambda c: c.in_)
    groups: List[List[Clip]] = [[ordered[0]]]
    for prev, nxt in zip(ordered, ordered[1:]):
        if _transition_between(transitions, prev.id, nxt.id):
            groups[-1].append(nxt)
        else:
            groups.append([nxt])
    for g in groups:
        for x, y in zip(g, g[1:]):
            assert _transition_between(transitions, x.id, y.id), \
                f"分组不变量破坏：组内 {x.id}->{y.id} 缺少 transition"
    for g1, g2 in zip(groups, groups[1:]):
        assert not _transition_between(transitions, g1[-1].id, g2[0].id), \
            f"分组不变量破坏：组间 {g1[-1].id}->{g2[0].id} 不应有 transition"
    return groups


def build_xfade_filter(label_a: str, label_b: str, out_label: str,
                       duration: float, offset: float,
                       transition_type: str = "fade") -> str:
    return (f"[{label_a}][{label_b}]xfade=transition={_xfade_kind(transition_type)}"
            f":duration={_fmt_num(duration)}:offset={_fmt_num(offset)}[{out_label}]")


def build_acrossfade_filter(label_a: str, label_b: str, out_label: str,
                            duration: float,
                            transition_type: str = "fade") -> str:
    """音频交叉淡化：曲线从 registry 取（tri = 线性）。

    注意：acrossfade 的 c1/c2 是**曲线类型**（nofade/tri/qsin/...），不接受
    "fade"（那是 xfade 的 transition 名）。tri 与 xfade=transition=fade 语义一致，
    且不向 planner 暴露曲线选择。
    """
    spec = resolve_transition(transition_type)
    curve = spec.audio_curve if spec else "tri"
    return (f"[{label_a}][{label_b}]acrossfade=d={_fmt_num(duration)}"
            f":c1={curve}:c2={curve}[{out_label}]")


def _audio_groups_for(v_groups: List[List[Clip]], vclips: List[Clip],
                      aclips: List[Clip]) -> Optional[List[List[Clip]]]:
    """把视频分组映射为音频分组；视频/音频数量不一致时返回 None（调用方降级）。"""
    if len(vclips) != len(aclips):
        return None
    v_idx = {c.id: i for i, c in enumerate(vclips)}
    out: List[List[Clip]] = []
    for g in v_groups:
        try:
            out.append([aclips[v_idx[c.id]] for c in g])
        except (KeyError, IndexError):
            return None
    return out


def _build_audio_chain_with_gaps(af_tail: List[str], base_labels: List[str],
                                 video_groups: List[List[Clip]]) -> List[str]:
    """构建带 gap 静音的音频链标签列表，与视频 tpad 黑帧同步。

    对应视频侧 T6 方案 A：组间 gap 用 tpad 补黑帧，音频侧同步插入等长静音，
    保证成片音画时长一致，不因 gap 被 `-shortest` 截断。

    实现要点：ffmpeg 里 filter 的输入/输出**不能同名**（`[x]atrim[x]` 会报标签重复），
    所以静音段用两个标签：`as{i}`（anullsrc 源）→ `asi{i}`（裁剪后）。
    """
    n_gaps = len(video_groups) - 1
    if n_gaps <= 0:
        return list(base_labels)

    gaps: List[float] = []
    for gi in range(n_gaps):
        gap = video_groups[gi + 1][0].in_ - video_groups[gi][-1].out
        gaps.append(gap if gap > 1e-6 else 0.0)

    extended: List[str] = []
    for i, lbl in enumerate(base_labels):
        if i > 0 and (i - 1) < n_gaps and gaps[i - 1] > 1e-6:
            src, dst = f"as{i - 1}", f"asi{i - 1}"
            af_tail.append(f"anullsrc=r=48000:cl=stereo[{src}]")
            af_tail.append(f"[{src}]atrim=start=0:end={_fmt_num(gaps[i - 1])},"
                           f"asetpts=PTS-STARTPTS[{dst}]")
            extended.append(f"[{dst}]")
        extended.append(lbl)
    return extended


def compile(tl: Timeline) -> Dict[str, Any]:
    """纯函数：只读 tl -> FFmpeg 描述。同输入同输出。

    转场与片段级操作的作用域（正交，互不影响）：
    - transform / effect 作用于片段的**源全段**（sourceIn..sourceOut），
      即进入 xfade 之前的完整画面；
    - transition 只在**片段边界的 duration 区间**内消费时间（xfade/acrossfade），
      被消耗的是边界时长，不会改变 transform / effect 已作用的内容。
    因此「链式 xfade 吃掉中间片段的两头」与「该片段上的调色/位移效果」互不干扰。
    """
    W = tl.metadata.resolution.get("w", 1920)
    H = tl.metadata.resolution.get("h", 1080)
    vclips = sorted([c for c in tl.clips if _track_type(tl, c) == "video"], key=lambda c: c.in_)
    aclips = sorted([c for c in tl.clips if _track_type(tl, c) == "audio"], key=lambda c: c.in_)
    unsupported: List[str] = []

    bgm_ids = {id(c) for c in aclips if _is_bgm(c)}
    main_aclips = [c for c in aclips if id(c) not in bgm_ids]
    bgm_clips = [c for c in aclips if id(c) in bgm_ids]

    # 转场（T5）：只保留「引用完整 + 时长合法」的；其余显式记 unsupported（不静默）
    valid_transitions = _valid_transitions(tl, unsupported)

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
                 "setpts=PTS-STARTPTS"]
        # T7：crop 在 scale 之前（源坐标系；w/h < 1 视为占比）
        crop = crop_filter(c.transform.crop)
        if crop:
            chain.append(crop)
        chain.append(f"scale={W}:{H}")
        # T7：变速必须在 trim/setpts 之后 —— 否则 trim 的 start/end 会按变速后的
        # PTS 解释，切点全错（把 setpts 插到链首会切空）。
        sr = _speed_rate(c)
        if sr is not None and abs(sr - 1.0) > 1e-9:
            chain.append(f"setpts=PTS/{_fmt_num(sr)}")
        # T4：几何变换（pad/rotate）在归一化之后、效果之前
        chain.extend(transform_geometry_filters(c, W, H))
        for e in c.effects or []:
            if (e or {}).get("effectType") in (BGM_MARKER, SPEED_MARKER):
                continue
            f = video_effect_filter(e, dur)
            if f:
                chain.append(f)
            else:
                unsupported.append(
                    f"画面效果 `{(e or {}).get('effectType')}` 不支持，已忽略"
                    f"（可用：{list(VIDEO_EFFECTS)}）")
        # T4：透明度最后应用（先调色/效果，再设 alpha）
        chain.extend(transform_opacity_filters(c))
        # T5：转场不再用片段级 fade 近似，改由汇总阶段的 xfade 完成
        vid_parts.append(f"[{i}:v]" + ",".join(chain) + f"[{lbl}]")
        vlabels.append(f"[{lbl}]")

    # T6：提前算视频分组与 gap —— 音频侧据此决定是否统一通道布局
    video_groups = group_clips_by_transition(vclips, valid_transitions)
    has_gap = any(
        video_groups[i + 1][0].in_ - video_groups[i][-1].out > 1e-6
        for i in range(len(video_groups) - 1))

    # ---- 主音频逐片段链 ----
    for j, c in enumerate(main_aclips):
        idx = len(vclips) + j
        inputs.append(c.sourceRef)
        lbl = f"ca{j}"
        dur = c.sourceOut - c.sourceIn
        chain = [f"atrim=start={c.sourceIn}:end={c.sourceOut}",
                 "asetpts=PTS-STARTPTS"]
        if has_gap:
            # 有 gap 要插静音段：先统一到 stereo。实测 mono/stereo 混拼不会报错，
            # 但会**静默下混成 mono**（输出 channels=1），属于隐性音质损失。
            chain.append("aformat=sample_rates=48000:channel_layouts=stereo")
        for e in c.effects or []:
            if (e or {}).get("effectType") == SPEED_MARKER:
                continue                       # speed 不是音频效果，单独处理
            f = audio_effect_filter(e, dur)
            if f:
                chain.append(f)
            else:
                unsupported.append(
                    f"音频效果 `{(e or {}).get('effectType')}` 不支持，已忽略"
                    f"（可用：{list(AUDIO_EFFECTS)}）")
        # T7：音频变速（effects 之后）
        sr_a = _speed_rate(c)
        if sr_a is not None:
            chain.extend(_atempo_filters(sr_a))
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

    # ---- 视频拼接（T5）：按转场分组；组内链式 xfade，组间 concat ----
    vmap = ""
    # video_groups 已在音频链之前算好（供 has_gap 判断），这里直接复用
    label_of = {c.id: f"cv{i}" for i, c in enumerate(vclips)}
    if len(vclips) == 1:
        vf_tail.append(f"{vlabels[0]}null[vout]")
        vmap = "[vout]"
    elif len(vclips) > 1:
        group_labels: List[str] = []
        for gi, g in enumerate(video_groups):
            if len(g) == 1:
                group_labels.append(f"[{label_of[g[0].id]}]")
                continue
            cur = label_of[g[0].id]
            acc = _stream_dur(g[0])
            for step, nxt in enumerate(g[1:], 1):
                prev = g[step - 1]
                tr = _transition_between(valid_transitions, prev.id, nxt.id)
                d = float(tr.duration)
                # 链式 xfade 的 offset 基准是「累积流时长」，不是单段时长
                offset = acc - d
                out = f"xg{gi}_{step}"
                vf_tail.append(build_xfade_filter(
                    cur, label_of[nxt.id], out, d, offset, tr.type))
                cur = out
                acc = acc + _stream_dur(nxt) - d
            group_labels.append(f"[{cur}]")
        # T6 方案 A：组间 gap 用 tpad 补黑帧；组内（含转场组内部）一律不插
        for gi in range(len(video_groups) - 1):
            gap = video_groups[gi + 1][0].in_ - video_groups[gi][-1].out
            if gap > 1e-6:
                lbl = f"gp{gi}"
                vf_tail.append(
                    f"{group_labels[gi]}tpad=stop_mode=add:"
                    f"stop_duration={_fmt_num(gap)}:color=black[{lbl}]")
                group_labels[gi] = f"[{lbl}]"
        if len(group_labels) == 1:
            vf_tail.append(f"{group_labels[0]}null[vout]")
        else:
            vf_tail.append("".join(group_labels)
                           + f"concat=n={len(group_labels)}:v=1:a=0[vout]")
        vmap = "[vout]"

    # ---- 文字水印（T7）：在字幕之前叠加 → 字幕保持在最上层 ----
    for idx, t in enumerate(tl.texts):
        if vmap:
            lbl = f"vtext{idx}"
            vf_tail.append(f"{vmap}{drawtext_filter(t, unsupported)}[{lbl}]")
            vmap = f"[{lbl}]"

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

    # ---- 主音频拼接（T5）：与视频同构分组时用 acrossfade，否则显式降级 ----
    # 无 BGM 时保持历史行为：主音频直接落 [aout]（对外契约不变）。
    final_label = "[amain]" if bgm else "[aout]"
    amain = ""
    if len(main_aclips) == 1:
        amain = alabels[0]
    elif len(main_aclips) > 1:
        audio_groups = _audio_groups_for(video_groups, vclips, main_aclips)
        if audio_groups is None:
            unsupported.append(
                f"音频片段数({len(main_aclips)})与视频({len(vclips)})不一致，"
                f"音频转场已跳过（成片可能音画不同步）")
            af_tail.append("".join(alabels)
                           + f"concat=n={len(main_aclips)}:v=0:a=1{final_label}")
            amain = final_label
        elif all(len(g) == 1 for g in audio_groups):
            # T6：组间 gap 音频侧同步插静音；无 gap 时与历史产出完全一致
            audio_chain = _build_audio_chain_with_gaps(af_tail, alabels, video_groups)
            af_tail.append("".join(audio_chain)
                           + f"concat=n={len(audio_chain)}:v=0:a=1{final_label}")
            amain = final_label
        else:
            a_index = {c.id: f"ca{j}" for j, c in enumerate(main_aclips)}
            a_group_labels: List[str] = []
            for gi, g in enumerate(audio_groups):
                vg = video_groups[gi]      # 同构视频组：transition 记的是视频 clip id
                if len(g) == 1:
                    a_group_labels.append(f"[{a_index[g[0].id]}]")
                    continue
                cur = a_index[g[0].id]
                for step, nxt in enumerate(g[1:], 1):
                    # 与视频取同一个 transition.duration —— 音画转场强制同步
                    tr = _transition_between(valid_transitions,
                                             vg[step - 1].id, vg[step].id)
                    assert tr is not None, "音频分组与视频分组不同构"
                    d = float(tr.duration)
                    out = f"ag{gi}_{step}"
                    af_tail.append(build_acrossfade_filter(
                        cur, a_index[nxt.id], out, d, tr.type))
                    cur = out
                a_group_labels.append(f"[{cur}]")
            # T6：组间 gap 音频侧同步插静音
            audio_chain = _build_audio_chain_with_gaps(
                af_tail, a_group_labels, video_groups)
            if len(audio_chain) == 1:
                af_tail.append(f"{audio_chain[0]}anull{final_label}")
            else:
                af_tail.append("".join(audio_chain)
                               + f"concat=n={len(audio_chain)}:v=0:a=1{final_label}")
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
