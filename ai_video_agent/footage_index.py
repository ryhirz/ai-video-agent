"""Footage Index 数据模型（M1/M2 共享）—— 喂给大模型的「素材上下文」，不是 Timeline。

为什么要单独建结构？
- Timeline 是「用户最终要编辑的时间轴」（契约 v1.0.0 的唯一真相）；
- Footage Index 是「素材理解结果」（ffprobe 元数据 / YOLOv8 视觉标签 / FunASR 转写），
  Agent 的 Retriever 只读取它来做规划，绝不会把整段视频喂给大模型（省成本、提稳定）。
对应 PRD §7.2：素材 → 结构化 Footage Index → 作为 LLM 上下文。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class VisualSegment:
    """一段带时间戳的视觉标签，例如 {label:'cat', start:12.3, end:15.7}。"""
    label: str                 # 类别：cat / person / speaker_closeup ...
    start: float               # 起始秒
    end: float                 # 结束秒
    confidence: float = 1.0    # 检测置信度
    source: str = "manual"     # yolo / mediapipe / manual
    bbox: List[float] = field(default_factory=list)  # [x1,y1,x2,y2] 归一化坐标（可选）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisualSegment":
        return cls(**d)


@dataclass
class TranscriptSegment:
    """一段带时间戳的转写，例如 {start:3.1, end:5.4, text:'大家好欢迎来到...'}。"""
    start: float
    end: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TranscriptSegment":
        return cls(**d)


@dataclass
class FootageIndex:
    mediaId: str
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)            # ffprobe：duration/fps/宽高/编码
    scenes: List[Dict[str, Any]] = field(default_factory=list)        # 镜头切分 [{start,end}]
    keyframes: List[float] = field(default_factory=list)              # 关键帧时间戳（M2 检测用）
    visual_segments: List[VisualSegment] = field(default_factory=list)      # M2 填充
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)  # M2 填充（FunASR）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mediaId": self.mediaId,
            "filepath": self.filepath,
            "metadata": self.metadata,
            "scenes": self.scenes,
            "keyframes": self.keyframes,
            "visual_segments": [v.to_dict() for v in self.visual_segments],
            "transcript_segments": [t.to_dict() for t in self.transcript_segments],
        }

    def dumps(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FootageIndex":
        return cls(
            mediaId=d["mediaId"],
            filepath=d["filepath"],
            metadata=d.get("metadata", {}),
            scenes=d.get("scenes", []),
            keyframes=d.get("keyframes", []),
            visual_segments=[VisualSegment.from_dict(v) for v in d.get("visual_segments", [])],
            transcript_segments=[TranscriptSegment.from_dict(t) for t in d.get("transcript_segments", [])],
        )

    @classmethod
    def loads(cls, s: str) -> "FootageIndex":
        return cls.from_dict(json.loads(s))
