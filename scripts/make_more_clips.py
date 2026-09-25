"""生成第二批测试素材（两段），与 `测试素材.mp4` 形成互补，专攻它覆盖不到的场景。

背景：现有 `测试素材.mp4` 是 1280x720 / 15s / 3 幕（bus → 人物 → bus）+ 中文旁白，
只能覆盖"两个 bus 区间 + 横屏 + 有语音"这一种形态。本脚本补两段：

------------------------------------------------------------------
素材2 `测试素材2_多目标.mp4`  1280x720@25fps / 24s / **6 幕**（bus / 人物 交替）
    旁白与幕严格对齐（每幕 0.25s 处起念，幕尾留 ~1.5s 静音）。
    覆盖场景：
      ① 多区间删除 —— 3 个 bus 区间（0-4 / 8-12 / 16-20s）。比原素材的 2 个更狠，
         正好打死"让模型自己推演链式派生 id"那类 bug（见 Bug J）。
      ② A/V 同步溯源 —— 旁白与幕一一对齐，silencedetect 的静音位置就是剪辑锚点。
      ③ YOLO 多目标（bus + person + tie 反复出现）。
      ④ 六幕跳切 —— 场景检测应能检出 5 个切换点。

------------------------------------------------------------------
素材3 `测试素材3_竖屏音乐.mp4`  720x1280@25fps (9:16) / 20s / **5 幕 + 真实推镜**
    音轨是 numpy 合成的纯音乐（**无语音**），全程有声、**无静音段**。
    覆盖场景：
      ① 竖屏分辨率 —— 缩放/留边/字幕排版的另一条路径。
      ② ASR 无语音的降级分支（原素材永远有旁白，覆盖不到）。
      ③ 连续音频（silencedetect 找不到静音）→ 音频溯源只能靠互相关，是压力测试。
      ④ 画面真在动（zoompan 推镜）→ 原素材是纯静态幻灯片，覆盖不到运动画面。

依赖：ffmpeg（tools/ffmpeg/bin）、numpy、edge-tts（venv 已装）。
真实图源：ultralytics 自带 assets（bus.jpg 竖图 810x1080 / zidane.jpg 横图 1280x720）。
幕间一律**硬切**（不加淡入淡出），理由见 `build_silent` 里的注释。
"""

from __future__ import annotations

import asyncio
import subprocess
import wave
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tests" / ".cache"
ASSETS = ROOT / ".venv" / "Lib" / "site-packages" / "ultralytics" / "assets"
FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"

BUS = ASSETS / "bus.jpg"
ZIDANE = ASSETS / "zidane.jpg"

OUT2 = ROOT / "测试素材2_多目标.mp4"
OUT3 = ROOT / "测试素材3_竖屏音乐.mp4"

SR = 44100


def run(cmd: Sequence[str]) -> None:
    """执行外部命令，失败即抛（避免"静默产出坏素材"）。"""
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


# --------------------------------------------------------------------------
# 视频：把若干幕静态图拼成一段带淡入淡出的无声视频
# --------------------------------------------------------------------------
def build_silent(acts: Sequence[Tuple[Path, float, bool]], out: Path,
                 w: int, h: int, fps: int, motion: bool) -> None:
    """acts = [(图片, 该幕秒数, 该幕是否推镜)]，拼成无声视频。

    每个输入都显式 `-loop 1 -t SEC` 限长：否则 ffmpeg 会等无限循环的图片输入，
    导致进程永不结束（上一版素材生成脚本踩过这个坑）。
    """
    inputs: List[str] = []
    for img, sec, _m in acts:
        inputs += ["-loop", "1", "-t", f"{sec}", "-i", str(img)]

    parts: List[str] = []
    tags: List[str] = []
    for i, (_img, sec, use_motion) in enumerate(acts):
        tag = f"v{i}"
        chain = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={fps},format=yuv420p"
        )
        if motion and use_motion:
            # zoompan 推镜：d=1 表示"每个输入帧出 1 帧"，配合 -loop 1 的输入
            # 逐帧累积 zoom（状态量），得到平滑的慢推效果。
            chain += (
                f",zoompan=z='min(1.0+0.0025*on,1.30)':d=1:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps}"
            )
        # 幕间用**硬切**，刻意不加淡入淡出。
        # 实测教训：每幕首尾各加 0.25s 淡入淡出后，跳切处会先"淡到黑"再"淡入"，
        # 而 ingest.scene_cuts 是 1Hz 采样直方图比对 —— 一个切点会被算成**两个**
        # （黑帧 vs 前幕、黑帧 vs 后幕各一次），6 幕的素材被判成 12 个场景；
        # 更糟的是落在黑帧上的关键帧喂给 YOLO 什么都检不出，白白污染视觉理解结果。
        # 硬切既符合真实素材形态，又能让场景检测得到干净的一对一边界。
        parts.append(f"[{i}:v]{chain}[{tag}]")
        tags.append(f"[{tag}]")

    fc = ";".join(parts) + ";" + "".join(tags) + f"concat=n={len(acts)}:v=1:a=0[outv]"
    run([FFMPEG, *inputs, "-filter_complex", fc, "-map", "[outv]",
         "-r", str(fps), "-pix_fmt", "yuv420p", "-an", "-y", out])


# --------------------------------------------------------------------------
# 音频 A：中文旁白（edge-tts，逐幕生成后按幕起点延时混入）
# --------------------------------------------------------------------------
async def _tts_one(text: str, voice: str, out: Path) -> None:
    import edge_tts
    comm = edge_tts.Communicate(text, voice=voice)
    await asyncio.wait_for(comm.save(str(out)), timeout=30)


def build_narration_tracks(lines: Sequence[str], voice: str = "zh-CN-XiaoxiaoNeural") -> List[Path]:
    """逐句合成旁白；任一句失败即整体失败（由调用方决定降级）。"""
    paths: List[Path] = []
    for i, text in enumerate(lines, 1):
        p = CACHE / f"_clip2_line{i}.mp3"
        asyncio.run(_tts_one(text, voice, p))
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"第 {i} 句旁白合成失败：{text}")
        paths.append(p)
    return paths


def mux_narration(silent: Path, tracks: Sequence[Path], starts: Sequence[float],
                  total: float, out: Path) -> None:
    """把每句旁白按 (幕起点 + 0.25s) 延时后**并行**叠加（这里 amix 才是正确用法：
    各句在不同时刻发声、需要同时存在；顺序拼接场景才该用 concat）。"""
    inputs: List[str] = ["-i", str(silent)]
    for t in tracks:
        inputs += ["-i", str(t)]

    parts: List[str] = []
    labels: List[str] = []
    for i, st in enumerate(starts):
        ms = int(round((st + 0.25) * 1000))
        parts.append(f"[{i + 1}:a]aresample={SR},adelay={ms}|{ms}[n{i}]")
        labels.append(f"[n{i}]")
    mix = ("".join(labels) + f"amix=inputs={len(tracks)}:duration=longest:"
           f"dropout_transition=0:normalize=0[mixed];[mixed]apad[aout]")
    fc = ";".join(parts) + ";" + mix

    run([FFMPEG, *inputs, "-filter_complex", fc,
         "-map", "0:v", "-map", "[aout]",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-t", f"{total}", "-y", out])


# --------------------------------------------------------------------------
# 音频 B：合成一段纯音乐（无语音、全程有声、无静音段）
# --------------------------------------------------------------------------
def synth_music(path: Path, seconds: float = 20.0) -> None:
    """五声音阶 + 拨弦包络 + 每拍低音鼓点。写 16bit 单声道 WAV。

    为什么用 numpy 而不是 ffmpeg 的 aevalsrc：表达式里的 `mod(t,0.5)` 含逗号，
    在 filtergraph 里要转义，且可读性差；直接合成波形更可控。
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR

    # 每 0.5s 一个音；五声音阶（半音偏移）上下行，听感不刺耳
    step_semitone = np.array([0, 2, 4, 7, 4, 2, 0, 4])   # 音高走向
    idx = (np.floor(t * 2).astype(np.int64)) % len(step_semitone)
    note = 220.0 * 2 ** (step_semitone[idx] / 12.0)

    # 拨弦包络：每 0.5s 起一个新的衰减（音头清晰 → 频谱有节拍感）
    env = np.exp(-4.0 * (t % 0.5) / 0.5)

    tone = (np.sin(2 * np.pi * note * t)
            + 0.45 * np.sin(2 * np.pi * 2 * note * t)
            + 0.20 * np.sin(2 * np.pi * 3 * note * t))
    melody = 0.22 * env * tone
    bass = 0.16 * np.sin(2 * np.pi * 110 * t) * np.exp(-7.0 * (t % 0.5))
    sig = melody + bass

    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def mux_music(silent: Path, music: Path, total: float, out: Path) -> None:
    run([FFMPEG, "-i", str(silent), "-i", str(music),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
         "-b:a", "128k", "-t", f"{total}", "-y", out])


# --------------------------------------------------------------------------
def probe(path: Path) -> str:
    r = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries",
         "format=duration", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True)
    return r.stdout.strip()


def make_clip2() -> None:
    """素材2：6 幕横屏 + 逐幕对齐的中文旁白。"""
    print("\n===== 生成 素材2 测试素材2_多目标.mp4 =====")
    acts = [(BUS, 4.0, False), (ZIDANE, 4.0, False), (BUS, 4.0, False),
            (ZIDANE, 4.0, False), (BUS, 4.0, False), (ZIDANE, 4.0, False)]
    silent = CACHE / "_clip2_silent.mp4"
    build_silent(acts, silent, w=1280, h=720, fps=25, motion=False)

    lines = [
        "第一幕，画面里有一辆公交车。",
        "第二幕，画面里是两个人。",
        "第三幕，又出现了一辆公交车。",
        "第四幕，还是这两个人。",
        "第五幕，公交车再次出现。",
        "第六幕，画面回到人物。",
    ]
    starts = [i * 4.0 for i in range(6)]
    total = 24.0
    try:
        tracks = build_narration_tracks(lines)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 旁白合成失败（可能网络问题），改为无声素材：{e}")
        run([FFMPEG, "-i", str(silent), "-c:v", "copy", "-t", f"{total}", "-y", OUT2])
    else:
        mux_narration(silent, tracks, starts, total, OUT2)
        for p in tracks:
            p.unlink(missing_ok=True)
    silent.unlink(missing_ok=True)
    print(f"[done] {OUT2}\n{probe(OUT2)}")


def make_clip3() -> None:
    """素材3：5 幕竖屏 + 推镜 + 纯音乐（无语音）。"""
    print("\n===== 生成 素材3 测试素材3_竖屏音乐.mp4 =====")
    acts = [(BUS, 4.0, True), (ZIDANE, 4.0, True), (BUS, 4.0, True),
            (ZIDANE, 4.0, True), (BUS, 4.0, True)]
    silent = CACHE / "_clip3_silent.mp4"
    build_silent(acts, silent, w=720, h=1280, fps=25, motion=True)

    music = CACHE / "_clip3_music.wav"
    synth_music(music, seconds=20.0)
    mux_music(silent, music, 20.0, OUT3)

    silent.unlink(missing_ok=True)
    music.unlink(missing_ok=True)
    print(f"[done] {OUT3}\n{probe(OUT3)}")


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    make_clip2()
    make_clip3()
    print("\n[all done] 两段测试素材已生成。")


if __name__ == "__main__":
    main()
