"""Timeline + Edit Schema 可执行契约（冻结 v1.0.0）— 数据模型层。

对应规格书 §1。clip 的字段命名为 in_（避免与 Python 关键字冲突），
序列化时自动映射回 "in"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal

TrackType = Literal["video", "audio", "subtitle", "effect"]


@dataclass
class Metadata:
    schemaVersion: str = "1.0.0"
    fps: int = 30
    resolution: Dict[str, int] = field(default_factory=lambda: {"w": 1920, "h": 1080})
    duration: float = 0.0


@dataclass
class Track:
    id: str
    type: TrackType


@dataclass
class Transform:
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0


@dataclass
class Clip:
    id: str
    track: str
    sourceRef: str
    in_: float          # 时间轴入点（秒）
    out: float          # 时间轴出点（秒）
    sourceIn: float     # 源素材入点（秒）—— 编译器用来生成 -ss
    sourceOut: float    # 源素材出点（秒）—— 编译器用来生成 -to
    transform: Transform = field(default_factory=Transform)
    effects: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["in"] = d.pop("in_")          # 还原为规格书里的 "in"
        d["transform"] = asdict(self.transform)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Clip":
        d = dict(d)
        if "in" in d:
            d["in_"] = d.pop("in")
        d["transform"] = Transform(**d.get("transform", {}))
        return cls(**d)


@dataclass
class Subtitle:
    id: str
    track: str
    text: str
    start: float
    end: float
    style: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    id: str
    type: str
    duration: float
    attachedClips: List[str] = field(default_factory=list)


@dataclass
class Timeline:
    metadata: Metadata = field(default_factory=Metadata)
    tracks: List[Track] = field(default_factory=list)
    clips: List[Clip] = field(default_factory=list)
    subtitles: List[Subtitle] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "tracks": [asdict(t) for t in self.tracks],
            "clips": [c.to_dict() for c in self.clips],
            "subtitles": [asdict(s) for s in self.subtitles],
            "transitions": [asdict(t) for t in self.transitions],
        }

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Timeline":
        return cls(
            metadata=Metadata(**d.get("metadata", {})),
            tracks=[Track(**t) for t in d.get("tracks", [])],
            clips=[Clip.from_dict(c) for c in d.get("clips", [])],
            subtitles=[Subtitle(**s) for s in d.get("subtitles", [])],
            transitions=[Transition(**t) for t in d.get("transitions", [])],
        )

    @classmethod
    def loads(cls, s: str) -> "Timeline":
        return cls.from_dict(json.loads(s))

    # ---- 不可变拷贝（apply 阶段用，保证确定性/可回放） ----
    def copy(self) -> "Timeline":
        return Timeline.loads(self.dumps())
