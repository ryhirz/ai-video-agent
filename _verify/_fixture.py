"""验收脚本共用的素材夹具（`_verify/clip6.mp4`）。

**为什么要有这个模块**：`verify_ops.py` 原本自己带一份 `ensure_fixture()`，
而 `verify_semantics.py` 直接假定素材已存在 —— 于是后者**单独复跑会崩**
（探测一个还没生成的 `_verify/clip6.mp4`），只有"先跑过 verify_ops 再跑它"才侥幸能过。
验收脚本的价值就在于**随时可复跑**，所以这份生成逻辑必须是共享的、且谁都能先调到。

素材规格：6s / 640x360 / **前 3 秒纯蓝 + 后 3 秒纯红** + 440Hz 正弦音轨。
纯色分幕是为了让"像素是否变化"一目了然 —— 灰度会让 R=G=B、转场会让边界变黑。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "_verify" / "clip6.mp4"
FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"


def ensure_fixture(verbose: bool = True) -> Path:
    """确保验证素材存在（不入库，首次运行自动生成）。幂等：已存在就直接返回。"""
    if CLIP.exists():
        return CLIP
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(FFMPEG), "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=3",
         "-f", "lavfi", "-i", "color=c=red:s=640x360:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(CLIP)],
        check=True, capture_output=True, text=True)
    if verbose:
        print(f"已生成验证素材: {CLIP}")
    return CLIP


if __name__ == "__main__":
    print(ensure_fixture())
