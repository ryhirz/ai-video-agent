"""M2 真 FunASR(Paraformer) 离线验证脚本。

用法（在 ai-video-agent 目录）：
    .venv/Scripts/python.exe scripts/paraformer_demo.py

流程：摄取带音轨测试片 -> understand.transcribe() 跑 Paraformer -> 打印逐句转写，
并与合成时用的原文做字符级匹配率，证明 ASR 整链在本环境实跑通过。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_video_agent.ingest import build_footage_index
from ai_video_agent.understand import transcribe

VIDEO = "tests/.cache/sample_with_audio.mp4"
EXPECT = "今天天气真好，我们一起来剪辑这段视频，把精彩的片段保留下来。"


def main() -> None:
    idx = build_footage_index(VIDEO)
    print("has_audio:", idx.metadata.get("has_audio"),
          "duration:", round(idx.metadata.get("duration", 0), 2),
          "audio_codec:", idx.metadata.get("audio_codec"))
    assert idx.metadata.get("has_audio"), "测试片必须含音轨"

    segs = transcribe(idx, device="cpu")
    print("=== TRANSCRIPT ===")
    for s in segs:
        print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")

    text = "".join(s.text for s in segs)
    print("=== EXPECT ===", EXPECT)
    ratio = sum(1 for a, b in zip(text, EXPECT) if a == b) / max(len(EXPECT), 1)
    print(f"MATCH_RATIO={ratio:.2%}")
    print("PARAFORMER_E2E_OK" if segs else "PARAFORMER_EMPTY")


if __name__ == "__main__":
    main()
