"""A/B 实测：VAD 分段转写 vs 整片一次转写，的耗时与结果对比。

分段让时间戳精度从"整片一条"提升到"半句级"，但代价是**每个语音段各调一次 ASR**
（素材2 六幕 → 11 段 → 11 次调用 + 11 次 ffmpeg 切音频）。这里量化这个代价，
判断是否值得做批量优化。

`transcribe(use_vad=...)` 正好提供了 A/B 开关，不用另写实现。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index
from ai_video_agent.understand import transcribe

CLIP = ROOT / "测试素材2_多目标.mp4"


def timed(fn, *a, **kw):
    t0 = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t0


def main() -> None:
    idx = build_footage_index(str(CLIP))

    print("预热（加载 Paraformer + VAD 权重）…")
    transcribe(idx, use_vad=False)
    transcribe(idx)                      # 顺便把 VAD 权重也热起来

    rows = []
    for label, kw in (("整片一次", {"use_vad": False}), ("VAD 分段", {"use_vad": True})):
        segs, dt = timed(transcribe, idx, **kw)
        rows.append({"mode": label, "seconds": round(dt, 2), "segments": len(segs)})
        print(f"\n== {label}：耗时 {dt:.2f}s，{len(segs)} 段")
        for s in segs[:4]:
            print(f"   [{s.start:.2f}-{s.end:.2f}] {s.text!r}")
        if len(segs) > 4:
            print(f"   …（共 {len(segs)} 段）")

    a, b = rows[0]["seconds"], rows[1]["seconds"]
    print(f"\n分段代价：{a:.2f}s → {b:.2f}s（{b / a:.2f}×）")

    (ROOT / "_verify" / "bench_transcribe.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
