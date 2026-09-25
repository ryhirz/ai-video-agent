"""对 4 个「疑似只改 Timeline、不影响成片」的操作做像素级/音频级判定。

判定思想：渲染两版——**带该操作** vs **不带该操作**，直接比对成片内容。
内容一致 => 该操作对成片无效（只是在 Timeline 里记了一笔）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
CLIP = ROOT / "_verify" / "clip6.mp4"
OUT = ROOT / "_verify" / "sem_out"
OUT.mkdir(parents=True, exist_ok=True)

# 素材由共用夹具生成（此前本脚本只假定它存在，单独复跑会崩在探测这一步）
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _verify 目录本身
from _fixture import ensure_fixture  # noqa: E402

ensure_fixture()


def base_tl():
    return build_initial_timeline(build_footage_index(str(CLIP)))


def render_ops(ops, tag: str):
    tl, _logs, _warns = apply_operations(base_tl(), ops, strict=True)
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    path, msg = render(tl, base_dir=str(d))
    if not path:
        raise RuntimeError(f"{tag} 渲染失败: {msg}")
    return path, tl


def frame_pixel(path: str, t: float):
    """取 t 秒处一帧，缩放成 1x1，返回 (r,g,b)。素材是纯色块，可直接代表画面。"""
    r = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", path, "-ss", f"{t:.3f}",
         "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    b = r.stdout
    if len(b) < 3:
        return None
    return (b[0], b[1], b[2])


def audio_pcm(path: str):
    r = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True)
    return r.stdout


def rms(pcm: bytes) -> float:
    import array
    if len(pcm) < 2:
        return -1.0
    a = array.array("h")
    a.frombytes(pcm[: len(pcm) // 2 * 2])
    if not len(a):
        return -1.0
    return (sum(x * x for x in a) / len(a)) ** 0.5


def pcm_diff(p1: bytes, p2: bytes) -> float:
    """两段 PCM 的平均绝对差（0 = 完全相同）。"""
    import array
    n = min(len(p1), len(p2)) // 2 * 2
    if n == 0:
        return -1.0
    a, b = array.array("h"), array.array("h")
    a.frombytes(p1[:n]); b.frombytes(p2[:n])
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


RESULTS = []


def report(name, verdict, detail):
    RESULTS.append({"op": name, "verdict": verdict, "detail": detail})
    print(f"[{'有效' if verdict else '无效'}] {name}")
    print(f"        {detail}")


# =========================================================================
# 基线：不做任何操作
# =========================================================================
base_path, _ = render_ops([], "base")
base_px_1s = frame_pixel(base_path, 1.0)
base_px_3s = frame_pixel(base_path, 3.0)
base_rms = rms(audio_pcm(base_path))
print(f"基线成片: t=1.0s 像素={base_px_1s}  t=3.0s 像素={base_px_3s}  音频RMS={base_rms:.1f}")
print("=" * 78)

# =========================================================================
# 1) effect（灰度）—— 若生效，蓝色像素 (0,0,255) 应变成 R=G=B
# =========================================================================
p, tl = render_ops([{"type": "effect",
                     "payload": {"targetClipId": "src_video",
                                 "effectType": "grayscale", "params": {}}}], "effect")
px = frame_pixel(p, 1.0)
is_gray = px and abs(px[0] - px[1]) < 12 and abs(px[1] - px[2]) < 12
report("effect(grayscale)", bool(is_gray) and px != base_px_1s,
       f"t=1.0s 像素: 基线{base_px_1s} -> 加效果后{px}（R=G=B 才说明灰度生效）")

# =========================================================================
# 2) transition（淡入淡出）—— 若生效，边界 t=3.0 应由纯红变成蓝红混合
# =========================================================================
p, tl = render_ops([{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
                    {"type": "transition",
                     "payload": {"transitionType": "fade", "duration": 0.8,
                                 "clips": ["src_video__a", "src_video__b"]}}], "transition")
px = frame_pixel(p, 3.0)
blended = px and not (abs(px[0] - base_px_3s[0]) < 12 and abs(px[2] - base_px_3s[2]) < 12)
report("transition(fade)", bool(blended),
       f"t=3.0s 像素: 基线{base_px_3s} -> 加转场后{px}（混合色才说明转场生效）"
       f"；Timeline 里 transitions={len(tl.transitions)}")

# =========================================================================
# 3) add_bgm —— 若真混音，音频应与基线不同；若被丢/当作顺序拼接，则与基线一致
# =========================================================================
p, tl = render_ops([{"type": "add_bgm",
                     "payload": {"sourceRef": str(CLIP), "in": 0.0, "out": 6.0,
                                 "volume": 0.5}}], "bgm")
bgm_rms = rms(audio_pcm(p))
d = pcm_diff(audio_pcm(p), audio_pcm(base_path))
report("add_bgm(混音)", d > 50,
       f"音频RMS: 基线{base_rms:.1f} -> 加BGM{bgm_rms:.1f}； "
       f"PCM 平均绝对差={d:.2f}（>50 才算真混进去了）")

# =========================================================================
# 4) move_clip —— 两种用法分别测
# =========================================================================
# 4a. 同轨调序（把后段 b 移到 0.0）
p, tl = render_ops([{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
                    {"type": "move_clip",
                     "payload": {"targetClipId": "src_video__b", "toIn": 0.0}}], "move_reorder")
order = [c.id for c in sorted([c for c in tl.clips if c.track == "v_main"],
                              key=lambda c: c.in_)]
px_after = frame_pixel(p, 1.0)
moved = px_after != base_px_1s
report("move_clip(同轨调序)", moved,
       f"Timeline 顺序={order}； t=1.0s 像素: 基线{base_px_1s} -> 移动后{px_after}"
       f"（顺序未变则画面不变）")

# 4b. 跨轨移动（v_main -> 新轨）
tl_test = base_tl()
from ai_video_agent.timeline import Track
tl_test.tracks.append(Track("v_extra", "video"))
try:
    tl2, _l, _w = apply_operations(tl_test, [{"type": "move_clip",
                                              "payload": {"targetClipId": "src_video",
                                                          "toTrack": "v_extra"}}],
                                   strict=True)
    moved_ok = tl2.clips[0].track == "v_extra"
except Exception as e:  # noqa: BLE001
    moved_ok = False
    print("   跨轨移动异常:", e)
report("move_clip(跨轨)", moved_ok, f"clip.track 是否改为 v_extra: {moved_ok}")

print("=" * 78)
n_ok = sum(1 for r in RESULTS if r["verdict"])
print(f"有效 {n_ok} / {len(RESULTS)}")
(OUT / "semantic_result.json").write_text(
    json.dumps({"base_px_1s": base_px_1s, "base_px_3s": base_px_3s,
                "base_rms": base_rms, "results": RESULTS},
               ensure_ascii=False, indent=2), encoding="utf-8")
