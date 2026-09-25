"""为「按台词删除」出样片到 `outputs/`（可复跑 demo）。

指令「删掉说公交车的片段」走**台词**路径（与视觉路径的 `样例_素材2_删掉三个bus.mp4` 结果一致，
但入口不同 —— 前者靠转写文本，后者靠 YOLO 标签）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.understand import detect_objects, transcribe
from ai_video_agent.agent import RuleBasedPlanner
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render

CLIP = ROOT / "测试素材2_多目标.mp4"
DEST = ROOT / "outputs" / "样例_素材2_删掉说公交车的片段.mp4"
NL = "删掉说公交车的片段"


def main() -> None:
    idx = build_footage_index(str(CLIP))
    idx.visual_segments = detect_objects(idx)
    idx.transcript_segments = transcribe(idx)

    tl = build_initial_timeline(idx)
    ops = RuleBasedPlanner().plan(NL, tl, idx)
    rng = [(o["payload"]["start"], o["payload"]["end"]) for o in ops]
    print(f"指令「{NL}」→ {len(ops)} 条 delete_range {rng}")
    tl2, _logs, warns = apply_operations(tl, ops, strict=True)

    out, msg = render(tl2, base_dir=str(ROOT / "_verify" / "speech_out"))
    if not out:
        print("渲染失败：", msg)
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out, DEST)
    kept = sorted((c.sourceIn, c.sourceOut) for c in tl2.clips if c.track == "v_main")
    print(f"保留片段 {kept}；警告：{warns or '无'}")
    print(f"样片已写到 {DEST}")


if __name__ == "__main__":
    main()
