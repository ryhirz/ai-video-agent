"""竖屏素材专项：`测试素材3_竖屏音乐.mp4` 走完整链路（摄取→加字幕→编译→渲染）。

为什么单独验：竖屏（720x1280）是本次新引入的形态，会同时影响
  - 编译期输出分辨率（取自 metadata.resolution）
  - 字幕 ASS 画布尺寸与定位（字幕会不会跑到画面外 / 被裁掉）
原素材全是 1280x720，覆盖不到这条路径。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline  # noqa: E402
from ai_video_agent.operations import apply_operations                        # noqa: E402
from ai_video_agent.render import render                                      # noqa: E402

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
OUT = ROOT / "_verify" / "mat_out"
OUT.mkdir(parents=True, exist_ok=True)

CLIP = ROOT / "测试素材3_竖屏音乐.mp4"


def probe_json(path) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "stream=codec_type,width,height", "-show_entries",
                        "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True, encoding="utf-8")
    return json.loads(r.stdout or "{}")


def main() -> None:
    idx = build_footage_index(str(CLIP))
    tl = build_initial_timeline(idx)
    meta = idx.metadata
    print(f"摄取 metadata: duration={meta.get('duration')} "
          f"{meta.get('width')}x{meta.get('height')} fps={meta.get('fps')} "
          f"has_audio={meta.get('has_audio')}")

    ops = [
        {"type": "add_subtitle",
         "payload": {"text": "竖屏测试字幕", "start": 0.5, "end": 3.5}},
        {"type": "trim", "payload": {"targetClipId": "src_video", "in": 2.0}},
    ]
    tl2, logs, warns = apply_operations(tl, ops, strict=True)
    print(f"应用操作: {len(ops)} 条  警告={warns}")

    out, msg = render(tl2, base_dir=str(OUT))
    if not out:
        print(f"❌ 渲染失败：{msg}")
        return

    info = probe_json(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    print(f"✅ 出片 {out}")
    print(f"   成片 分辨率={v.get('width')}x{v.get('height')}  "
          f"时长={float(info['format']['duration']):.3f}s（源 20s，去掉前 2s → 期望 18s）")

    # 抽字幕所在时刻的帧，肉眼确认没跑出画面
    frame = OUT / "vertical_subtitle.png"
    subprocess.run([str(FFMPEG), "-v", "error", "-ss", "1.0", "-i", str(out),
                    "-frames:v", "1", "-y", str(frame)], check=True)
    print(f"   抽帧（t=1.0s，字幕应在画面内）: {frame}")
    print(f"   竖向是否正确: {v.get('height', 0) > v.get('width', 0)}")


if __name__ == "__main__":
    main()
