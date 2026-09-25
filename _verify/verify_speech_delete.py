"""端到端实测「按台词删除」：素材2 上跑两条指令，真出片并 ffprobe 校验。

- 「删掉说公交车的片段」→ 走**台词**检索（`_speech_ranges`，区间外扩到镜头）
- 「删掉有公交车的片段」→ 走**视觉**检索（YOLO 的 bus 标签）

两条指令在素材2 上应给出**相同的三个镜头**（0–4 / 8–12 / 16–20s）——
因为素材设计上"公交幕"的旁白里正好也说了"公交车"。这个巧合正是交叉验证。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.understand import detect_objects, transcribe
from ai_video_agent.agent import RuleBasedPlanner, facts_block
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIP = ROOT / "测试素材2_多目标.mp4"
OUTDIR = ROOT / "_verify" / "speech_out"


def probe(path: str) -> dict:
    def q(args):
        return subprocess.run([str(FFPROBE), "-v", "error", *args, path],
                              capture_output=True, text=True).stdout.strip()
    return {"duration": float(q(["-show_entries", "format=duration", "-of", "csv=p=0"]) or 0),
            "video": q(["-select_streams", "v:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0"]),
            "audio": q(["-select_streams", "a:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0"])}


def main() -> None:
    idx = build_footage_index(str(CLIP))
    print("M2 视觉理解（YOLO + VAD 分段转写）…")
    idx.visual_segments = detect_objects(idx)
    idx.transcript_segments = transcribe(idx)
    print(f"  视觉段 {len(idx.visual_segments)} 条；转写段 {len(idx.transcript_segments)} 条")
    print("\n--- 事实块（交给 LLM 的那种） ---")
    print(facts_block(idx))

    results = []
    for nl in ("删掉说公交车的片段", "删掉有公交车的片段"):
        tl = build_initial_timeline(idx)
        ops = RuleBasedPlanner().plan(nl, tl, idx)
        rng = [(o["payload"]["start"], o["payload"]["end"]) for o in ops]
        print(f"\n=== 指令：{nl}\n  规划出 {len(ops)} 条 delete_range：{rng}")
        tl2, logs, warns = apply_operations(tl, ops, strict=True)
        out, msg = render(tl2, base_dir=str(OUTDIR))
        info = probe(out) if out else {}
        kept = sorted((c.sourceIn, c.sourceOut)
                      for c in tl2.clips if c.track == "v_main")
        print(f"  保留片段：{kept}")
        print(f"  渲染：{msg}")
        if out:
            print(f"  成片：{info['duration']:.2f}s (video={info['video']}, audio={info['audio']})")
        print(f"  警告：{warns or '无'}")
        results.append({"nl": nl, "ops": len(ops), "ranges": rng, "kept": kept,
                        "output": out, "probe": info, "warns": warns})

    (ROOT / "_verify" / "speech_delete.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n明细已写入 _verify/speech_delete.json")


if __name__ == "__main__":
    main()
