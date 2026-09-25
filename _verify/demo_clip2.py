"""用新素材出可直看的样片，并把上一轮那支样片按修复后的正确行为重出一版。

产出（写进 outputs/）：
  1. `样例_素材2_删掉三个bus.mp4`  —— 新素材2 跑「删掉有bus的片段」，3 个公交镜头全删（24s → 12s）
  2. `样例_删掉bus片段_加字幕.mp4` —— 旧素材重出：修复"只删关键帧跨度、留下同镜头残段"后，
                                     应为 15s → **5s**（只剩中间人物幕；旧版是 8s，残留 3s 公交画面）
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.agent import RuleBasedPlanner                       # noqa: E402
from ai_video_agent.ingest import build_footage_index, build_initial_timeline  # noqa: E402
from ai_video_agent.operations import apply_operations                  # noqa: E402
from ai_video_agent.render import render                                # noqa: E402
from ai_video_agent.understand import detect_objects                    # noqa: E402

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
OUTDIR = ROOT / "outputs"
TMP = ROOT / "_verify" / "mat_out"


def dur(path: str | Path) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return round(float(r.stdout.strip() or 0), 3)


def make(clip: Path, nl: str, extra: list | None, out_name: str) -> None:
    print(f"\n===== {clip.name} + 「{nl}」 =====")
    idx = build_footage_index(str(clip))
    idx.visual_segments = detect_objects(idx)
    tl = build_initial_timeline(idx)
    ops = RuleBasedPlanner().plan(nl, tl, idx) + (extra or [])
    tl2, _logs, warns = apply_operations(tl, ops, strict=True)

    v = sorted((round(c.sourceIn, 2), round(c.sourceOut, 2))
               for c in tl2.clips if c.track == "v_main")
    a = sorted((round(c.sourceIn, 2), round(c.sourceOut, 2))
               for c in tl2.clips if c.track == "a_main")
    print(f"  操作 {len(ops)} 条 / 警告 {warns}")
    print(f"  保留 视频={v}")
    print(f"     音频={a}   A/V 镜像={v == a}")

    path, msg = render(tl2, base_dir=str(TMP))
    if not path:
        print(f"  ❌ 渲染失败：{msg}")
        return
    dst = OUTDIR / out_name
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    print(f"  ✅ 源 {dur(clip):.3f}s → 成片 {dur(dst):.3f}s  →  {dst.name}")


def main() -> None:
    make(ROOT / "测试素材2_多目标.mp4", "删掉有bus的片段", None,
         "样例_素材2_删掉三个bus.mp4")
    make(ROOT / "测试素材.mp4", "删掉有bus的片段",
         [{"type": "add_subtitle",
           "payload": {"text": "开场", "start": 0.3, "end": 2.5}}],
         "样例_删掉bus片段_加字幕.mp4")


if __name__ == "__main__":
    main()
