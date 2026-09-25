"""M2 真实可用性验证：对真实素材跑 YOLOv8 目标检测 + FunASR 转写。

判定：必须产出**非空的真实结果**，不能是"依赖缺失 -> 静默降级成空列表"。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index
from ai_video_agent.understand import build_visual_understanding

CLIP = str(ROOT / "测试素材.mp4")
OUT = ROOT / "_verify" / "m2_result.json"


def main() -> None:
    print("[M1] 摄取...", flush=True)
    t0 = time.time()
    idx = build_footage_index(CLIP)
    print(f"     耗时 {time.time()-t0:.1f}s | 时长={idx.metadata['duration']}s "
          f"场景数={len(idx.scenes)} 关键帧={len(idx.keyframes)}", flush=True)

    print("[M2] 视觉理解（YOLOv8）+ 语音转写（FunASR）...", flush=True)
    t0 = time.time()
    idx = build_visual_understanding(idx)
    dt = time.time() - t0
    print(f"     耗时 {dt:.1f}s", flush=True)

    vis = idx.visual_segments or []
    tr = idx.transcript_segments or []

    print("\n===== YOLOv8 视觉段 =====")
    for s in vis:
        print(f"  {s}")
    print("\n===== FunASR 转写段 =====")
    for s in tr:
        print(f"  {s}")

    labels = set()
    for s in vis:
        labels.add(s.label)

    texts = " ".join((s.text or "") for s in tr)

    print("\n===== 判定 =====")
    print(f"  视觉段 {len(vis)} 条 / 目标类别: {sorted(x for x in labels if x)}")
    print(f"  转写段 {len(tr)} 条 / 文本: {texts.strip()!r}")
    print(f"  YOLO 可用: {len(vis) > 0 and bool(labels)}")
    print(f"  ASR 可用:  {len(tr) > 0 and bool(texts.strip())}")

    OUT.write_text(json.dumps({
        "clip": CLIP, "duration": idx.metadata["duration"],
        "scenes": idx.scenes, "keyframes": idx.keyframes,
        "visual_segments": [v.to_dict() for v in vis],
        "transcript_segments": [t.to_dict() for t in tr],
        "index_full": idx.to_dict(),
        "visual_ok": len(vis) > 0 and bool(labels),
        "asr_ok": len(tr) > 0 and bool(texts.strip()),
        "elapsed_s": round(dt, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细已写入 {OUT}")


if __name__ == "__main__":
    main()
