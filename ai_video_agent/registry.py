"""能力注册表（单一真相）：操作类型 / 效果 / 参数域。

设计约束
--------
1. **单一真相**：编译器（compiler）、操作校验（operations）、给 LLM 的能力清单
   （`capability_manifest`）全部从本模块取，禁止各自硬编码。
2. **清单由代码导出**：`capability_manifest()` 由本表生成，**不允许手写 JSON**——
   否则会重演「提示词说某能力可用、编译器却不认」的漂移。
3. **向后兼容**：对外暴露 `VIDEO_EFFECTS` / `AUDIO_EFFECTS` / `BGM_MARKER`，
   以及 `SIMPLE_VIDEO_EFFECTS` / `PARAM_VIDEO_EFFECTS`（与 compiler 旧接口同构），
   调用方（operations.py / tests）无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# 参数域
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    """一个参数的合法域。数值型 default 必须落在 [min, max] 内。"""

    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    description: str = ""
    type: str = "number"          # number | string（rotation 允许含 t 的表达式）

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.default is not None:
            d["default"] = self.default
        if self.min is not None:
            d["min"] = self.min
        if self.max is not None:
            d["max"] = self.max
        if self.description:
            d["description"] = self.description
        d["type"] = self.type
        return d


# ---------------------------------------------------------------------------
# 操作类型：与 operations.apply_operation 的 handlers 一一对应
# ---------------------------------------------------------------------------
OPERATIONS: Dict[str, Dict[str, Any]] = {
    "cut":          {"summary": "在时间轴某点切分片段", "payload": ["targetClipId", "at"]},
    "trim":         {"summary": "裁剪片段头/尾", "payload": ["targetClipId", "in", "out"]},
    "add_subtitle": {"summary": "添加字幕", "payload": ["text", "start", "end", "style"]},
    "add_bgm":      {"summary": "添加背景音乐", "payload": ["sourceRef", "in", "out",
                                                           "volume", "fadeIn", "fadeOut"]},
    "delete_clip":  {"summary": "删除整个片段", "payload": ["targetClipId"]},
    "delete_range": {"summary": "按源素材时间区间删除", "payload": ["sourceRef", "start", "end"]},
    "move_clip":    {"summary": "移动片段轨道/位置", "payload": ["targetClipId", "toTrack",
                                                                "toIn", "toIndex"]},
    "transition":   {"summary": "两片段之间的转场", "payload": ["transitionType", "duration",
                                                               "clips"]},
    "effect":       {"summary": "给片段挂效果", "payload": ["targetClipId", "effectType",
                                                           "params"]},
}

TRANSITION_TYPES: Tuple[str, ...] = ("fade", "cross", "wipe")


# ---------------------------------------------------------------------------
# transform（几何 / 透明度）：compiler 映射为 pad / rotate / colorchannelmixer
# ---------------------------------------------------------------------------
TRANSFORM_PARAMS: Dict[str, ParamSpec] = {
    "x": ParamSpec(0, None, None, "水平偏移像素（pad 的 x；ffmpeg pad 不支持负值）", "number"),
    "y": ParamSpec(0, None, None, "垂直偏移像素（pad 的 y；ffmpeg pad 不支持负值）", "number"),
    "scale": ParamSpec(1.0, None, None, "缩放倍数（1.0 = 原始尺寸）", "number"),
    "rotation": ParamSpec(0, None, None,
                          "旋转弧度（rad）；可传数值，或含 t 的表达式", "number"),
    "opacity": ParamSpec(1.0, 0.0, 1.0, "画面透明度", "number"),
}


def validate_transform(values: Dict[str, Any]) -> List[str]:
    """校验 transform 取值；返回错误列表（空 = 通过）。

    rotation 允许两种形态：数值（弧度）、或含 t 的表达式字符串（如 "0.5*t"）。
    其余字段必须是数字。
    """
    errs: List[str] = []
    v = values or {}
    for name, spec in TRANSFORM_PARAMS.items():
        raw = v.get(name, spec.default)
        if name == "rotation" and isinstance(raw, str):
            s = raw.strip()
            if not s:
                errs.append("transform.rotation 不能为空字符串")
            elif "t" not in s:
                try:
                    float(s)
                except ValueError:
                    errs.append(
                        f"transform.rotation 表达式必须含 t 或为数值：{raw!r}")
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errs.append(f"transform.{name} 必须是数字，收到 {raw!r}")
            continue
        if spec.min is not None and raw < spec.min:
            errs.append(f"transform.{name}={raw} 小于下限 {spec.min}")
        if spec.max is not None and raw > spec.max:
            errs.append(f"transform.{name}={raw} 超过上限 {spec.max}")
    return errs


# ---------------------------------------------------------------------------
# 效果规格
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EffectSpec:
    """单个效果的规格。

    kind:
      - "simple" 固定滤镜串（无参数）
      - "param"  参数化滤镜，`filter` 是含 {v} 的模板
      - "fade"   淡入/淡出，由 video_effect_filter / audio_effect_filter 特殊分支处理
    """

    canonical: str
    kind: str
    filter: str = ""
    param: Optional[ParamSpec] = None
    param_name: str = "value"
    aliases: Tuple[str, ...] = ()

    def names(self) -> Tuple[str, ...]:
        return (self.canonical,) + self.aliases

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "aliases": list(self.aliases)}
        if self.kind == "simple":
            d["filter"] = self.filter
        elif self.kind == "param":
            d["filter_template"] = self.filter
            d["params"] = {self.param_name: self.param.to_dict()}
        else:  # fade
            d["params"] = {"duration": self.param.to_dict()}
        return d


_SEPIA = "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"

# 画面效果（canonical -> spec）。范围取自契约 §2.6 与 FFmpeg 滤镜自身约束。
VIDEO_EFFECT_SPECS: Dict[str, EffectSpec] = {
    "grayscale":  EffectSpec("grayscale", "simple", "hue=s=0", aliases=("灰度", "黑白")),
    "sepia":      EffectSpec("sepia", "simple", _SEPIA, aliases=("复古",)),
    "invert":     EffectSpec("invert", "simple", "negate", aliases=("反色",)),
    "mirror":     EffectSpec("mirror", "simple", "hflip", aliases=("hflip", "水平翻转")),
    "vflip":      EffectSpec("vflip", "simple", "vflip", aliases=("垂直翻转",)),
    "blur":       EffectSpec("blur", "param", "{v}", ParamSpec(5, 0, 50),
                             param_name="radius", aliases=("模糊",)),
    "brightness": EffectSpec("brightness", "param", "brightness={v}", ParamSpec(0.1, -1, 1),
                             aliases=("亮度",)),
    "contrast":   EffectSpec("contrast", "param", "contrast={v}", ParamSpec(1.5, 0, 2),
                             aliases=("对比度",)),
    "saturation": EffectSpec("saturation", "param", "saturation={v}", ParamSpec(1.5, 0, 2),
                             aliases=("饱和度",)),
    "fade_in":    EffectSpec("fade_in", "fade", param=ParamSpec(None, 0, 60), aliases=("淡入",)),
    "fade_out":   EffectSpec("fade_out", "fade", param=ParamSpec(None, 0, 60), aliases=("淡出",)),
}

# 音频效果
AUDIO_EFFECT_SPECS: Dict[str, EffectSpec] = {
    "volume":   EffectSpec("volume", "param", "volume={v}", ParamSpec(1.0, 0, 2),
                           aliases=("音量",)),
    "fade_in":  EffectSpec("fade_in", "fade", param=ParamSpec(None, 0, 60), aliases=("淡入",)),
    "fade_out": EffectSpec("fade_out", "fade", param=ParamSpec(None, 0, 60), aliases=("淡出",)),
}

# add_bgm 在 clip 上打的内部标记（不是用户可见效果）
BGM_MARKER = "bgm_mix"


# ---------------------------------------------------------------------------
# 向后兼容的派生表（compiler 旧接口）
# ---------------------------------------------------------------------------
def _build_compat_tables(specs: Dict[str, EffectSpec]) -> Tuple[Dict[str, str], Dict[str, tuple]]:
    simple: Dict[str, str] = {}
    param: Dict[str, tuple] = {}
    for spec in specs.values():
        for name in spec.names():
            if spec.kind == "simple":
                simple[name] = spec.filter
            elif spec.kind == "param":
                # 与 compiler 原元组同构：(filter, param_name, default, template)
                param[name] = (spec.filter, spec.param_name, spec.param.default, spec.filter)
    return simple, param


SIMPLE_VIDEO_EFFECTS, PARAM_VIDEO_EFFECTS = _build_compat_tables(VIDEO_EFFECT_SPECS)
SIMPLE_AUDIO_EFFECTS, PARAM_AUDIO_EFFECTS = _build_compat_tables(AUDIO_EFFECT_SPECS)


def _all_names(specs: Dict[str, EffectSpec]) -> Tuple[str, ...]:
    names = set()
    for spec in specs.values():
        names.update(spec.names())
    return tuple(sorted(names))


VIDEO_EFFECTS: Tuple[str, ...] = _all_names(VIDEO_EFFECT_SPECS)
AUDIO_EFFECTS: Tuple[str, ...] = _all_names(AUDIO_EFFECT_SPECS)


# ---------------------------------------------------------------------------
# 能力清单（由代码导出，禁止手写）
# ---------------------------------------------------------------------------
def capability_manifest() -> Dict[str, Any]:
    """机器可读的能力清单：供 planner 提示词注入 + 一致性断言。"""
    return {
        "schema_version": MANIFEST_VERSION,
        "operations": sorted(OPERATIONS),
        "transition_types": list(TRANSITION_TYPES),
        "transform": {name: spec.to_dict() for name, spec in TRANSFORM_PARAMS.items()},
        "effects": {
            "video": {s.canonical: s.to_dict() for s in VIDEO_EFFECT_SPECS.values()},
            "audio": {s.canonical: s.to_dict() for s in AUDIO_EFFECT_SPECS.values()},
        },
        "internal_markers": [BGM_MARKER],
    }


def dump_manifest(path: str) -> str:
    """把能力清单写成 JSON 文件。

    这是 `capability_manifest.json` 的**唯一合法生成方式**——禁止手改，
    因为手改过的清单会与编译器脱节，正是本模块要消灭的那类漂移。
    """
    import json
    from pathlib import Path

    p = Path(path)
    p.write_text(
        json.dumps(capability_manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(p)


def resolve_effect(name: str, *, audio: bool = False) -> Optional[EffectSpec]:
    """按规范名或别名查效果规格；查不到返回 None（供校验层使用）。"""
    specs = AUDIO_EFFECT_SPECS if audio else VIDEO_EFFECT_SPECS
    for spec in specs.values():
        if name in spec.names():
            return spec
    return None
