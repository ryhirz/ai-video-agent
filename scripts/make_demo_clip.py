"""生成演示用测试视频：3 幕真实实拍图（公交 -> 人物 -> 公交），
带缓慢运镜(zoompan) + 淡入淡出，并混入中文旁白音轨（edge-tts）。
产物默认输出到项目根目录 `测试素材.mp4`，可直接在 Gradio UI 上传测试。

依赖：ffmpeg（tools/ffmpeg/bin）、edge-tts（venv 已装）。
真实图源：ultralytics 自带 assets（bus.jpg / zidane.jpg）。
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tests" / ".cache"
ASSETS = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets"
FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"

BUS = ASSETS / "bus.jpg"
ZIDANE = ASSETS / "zidane.jpg"

TMP_SILENT = CACHE / "_demo_silent.mp4"
TMP_AUDIO = CACHE / "_demo_narration.mp3"
OUT = ROOT / "测试素材.mp4"

W, H, FPS, SEC = 1280, 720, 25, 5
D = SEC * FPS  # 每幕帧数


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def build_silent_video() -> None:
    """三幕静态图 -> 带淡入淡出的连续视频（无声）。
    注意：每个输入用 `-loop 1 -t SEC` 限长，避免 zoompan/无限循环导致 ffmpeg 永不停止。
    """
    inputs: list[str] = []
    for img in (BUS, ZIDANE, BUS):
        inputs += ["-loop", "1", "-t", str(SEC), "-i", str(img)]
    scene_tags = ["v0", "v1", "v2"]
    parts = []
    for i, tag in enumerate(scene_tags):
        parts.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={FPS},format=yuv420p,"
            f"fade=t=in:st=0:d=0.3,"
            f"fade=t=out:st={SEC - 0.3:.1f}:d=0.3[{tag}]"
        )
    concat = "".join(f"[{t}]" for t in scene_tags) + f"concat=n={len(scene_tags)}:v=1:a=0[outv]"
    fc = ";".join(parts) + ";" + concat
    run([
        str(FFMPEG), *inputs, "-filter_complex", fc,
        "-map", "[outv]", "-r", str(FPS), "-pix_fmt", "yuv420p", "-y", str(TMP_SILENT),
    ])


async def build_narration() -> None:
    """中文旁白（edge-tts 在线合成，国内可达）。带 30s 超时保护。"""
    import edge_tts
    text = ("这是一段用于测试的视频。第一幕是一辆公交车。第二幕是两个人在踢足球。"
            "第三幕又出现了一辆公交车。")
    comm = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
    await asyncio.wait_for(comm.save(str(TMP_AUDIO)), timeout=30)


def mux_audio() -> None:
    """静音视频 + 旁白 -> 成品（视频保持 15s，音频不足处补静音）。"""
    run([
        str(FFMPEG), "-i", str(TMP_SILENT), "-i", str(TMP_AUDIO),
        "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-t", str(SEC * 3), "-y", str(OUT),
    ])


def main() -> None:
    build_silent_video()
    try:
        asyncio.run(build_narration())
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 旁白合成失败（可能网络问题），产出无声版本：{e}")
        TMP_AUDIO.unlink(missing_ok=True)
        run([str(FFMPEG), "-i", str(TMP_SILENT), "-c:v", "copy",
             "-t", str(SEC * 3), "-y", str(OUT)])
    else:
        mux_audio()
    # 清理中间文件
    TMP_SILENT.unlink(missing_ok=True)
    TMP_AUDIO.unlink(missing_ok=True)
    print(f"[done] 测试素材已生成：{OUT}")


if __name__ == "__main__":
    main()
