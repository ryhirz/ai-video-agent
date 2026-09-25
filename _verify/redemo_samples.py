"""把带字幕的样片全部按**当前**代码重出一遍。

背景：修掉「字幕在竖屏/小分辨率横向溢出被裁」之后，ASS 的坐标系与字号都变了，
所有用旧代码出的字幕样片都已过时（字号偏大、坐标系未声明）。这里统一重出，
顺带在横屏上确认修复没有把字号改坏。

产出（outputs/）：
  - `样例_加字幕.mp4`            横屏 15s + 字幕「开场」
  - `样例_删掉bus片段_加字幕.mp4` 横屏 5s（公交镜头全删）+ 字幕
  - `样例_字幕预览.png`          上片抽帧（横屏字号确认）
  - `验证_字幕单层.png`          同一条字幕下两遍 → 抽帧确认只有一层（无重影）
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

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
OUTDIR = ROOT / "outputs"
TMP = ROOT / "_verify" / "mat_out"
CLIP = ROOT / "测试素材.mp4"

SUB = {"type": "add_subtitle",
       "payload": {"text": "开场", "start": 0.3, "end": 3.0}}


def dur(p) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(p)], capture_output=True, text=True)
    return round(float(r.stdout.strip() or 0), 3)


def frame(video, t: float, dst: Path) -> None:
    subprocess.run([str(FFMPEG), "-v", "error", "-ss", str(t), "-i", str(video),
                    "-frames:v", "1", "-y", str(dst)], check=True)


def build(idx, tl, ops, name: str) -> Path:
    tl2, _logs, warns = apply_operations(tl, ops, strict=True)
    v = sorted((round(c.sourceIn, 2), round(c.sourceOut, 2))
               for c in tl2.clips if c.track == "v_main")
    path, msg = render(tl2, base_dir=str(TMP))
    assert path, msg
    dst = OUTDIR / name
    OUTDIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    print(f"  ✅ {name}  保留={v}  时长={dur(dst)}s  警告={warns}")
    return dst


def main() -> None:
    idx = build_footage_index(str(CLIP))
    idx.visual_segments = detect_objects(idx)
    tl = build_initial_timeline(idx)

    print("== 横屏 + 字幕 ==")
    v1 = build(idx, tl, [SUB], "样例_加字幕.mp4")
    frame(v1, 1.0, OUTDIR / "样例_字幕预览.png")
    print(f"  ✅ 样例_字幕预览.png（t=1.0s 抽帧）")

    print("\n== 删掉公交镜头 + 字幕 ==")
    ops = RuleBasedPlanner().plan("删掉有bus的片段", tl, idx) + [SUB]
    build(idx, tl, ops, "样例_删掉bus片段_加字幕.mp4")

    print("\n== 同一条字幕下两遍（去重 → 单层）==")
    v2 = build(idx, tl, [SUB, dict(SUB)], "样例_字幕去重验证.mp4")
    frame(v2, 1.0, OUTDIR / "验证_字幕单层.png")
    print("  ✅ 验证_字幕单层.png（t=1.0s 抽帧；应只有一层字，无重影）")


if __name__ == "__main__":
    main()
