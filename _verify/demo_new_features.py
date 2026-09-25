"""补齐 4 类操作渲染后的最终验收：在真实素材上出片，产出可直接查看的样片。

同时给出「操作前 vs 操作后」的像素/音频硬证据。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIP = str(ROOT / "测试素材.mp4")
OUT = ROOT / "_verify" / "after_out"
SAMPLES = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES.mkdir(parents=True, exist_ok=True)

BGM_FILE = OUT / "bgm_soft.wav"


def dur(p: str) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return -1.0


def pixel(p: str, t: float):
    r = subprocess.run([str(FFMPEG), "-v", "error", "-i", p, "-ss", f"{t:.3f}",
                        "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "-"], capture_output=True)
    b = r.stdout
    return (b[0], b[1], b[2]) if len(b) >= 3 else None


def pcm(p: str) -> bytes:
    return subprocess.run([str(FFMPEG), "-v", "error", "-i", p, "-vn", "-ac", "1",
                           "-ar", "16000", "-f", "s16le", "-"],
                          capture_output=True).stdout


def rms(b: bytes) -> float:
    import array
    n = len(b) // 2 * 2
    if not n:
        return -1.0
    a = array.array("h"); a.frombytes(b[:n])
    return (sum(x * x for x in a) / len(a)) ** 0.5


def main() -> None:
    base_tl = build_initial_timeline(build_footage_index(CLIP))
    print(f"素材: {CLIP} 时长={dur(CLIP):.2f}s")

    # 基线
    p_base, _ = render(base_tl, base_dir=str(OUT / "base"))
    assert p_base, "基线渲染失败"
    print(f"基线成片: {p_base}")

    # 造一段柔和的 BGM 素材（两个正弦叠加 + 淡入淡出），用于证明混音
    subprocess.run([str(FFMPEG), "-v", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=15",
                    "-f", "lavfi", "-i", "sine=frequency=277:duration=15",
                    "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0,volume=0.35[a]",
                    "-map", "[a]", "-c:a", "pcm_s16le", str(BGM_FILE)],
                   check=True, capture_output=True)
    print(f"BGM 素材: {BGM_FILE}")

    cases = [
        ("黑白", [{"type": "effect", "payload": {"targetClipId": "src_video",
                                                 "effectType": "grayscale", "params": {}}}],
         "样例_黑白效果.mp4", 1.0),
        ("淡入淡出转场", [
            {"type": "cut", "payload": {"targetClipId": "src_video", "at": 4.0}},
            {"type": "transition", "payload": {"transitionType": "fade", "duration": 0.6,
                                               "clips": ["src_video__a", "src_video__b"]}}],
         "样例_淡入淡出转场.mp4", 4.0),
        ("背景音乐混音", [{"type": "add_bgm", "payload": {"sourceRef": str(BGM_FILE),
                                                          "in": 0.0, "out": 15.0,
                                                          "volume": 0.6,
                                                          "fadeIn": 1.0, "fadeOut": 1.0}}],
         "样例_背景音乐混音.mp4", 1.0),
        ("片段重排", [
            {"type": "cut", "payload": {"targetClipId": "src_video", "at": 4.0}},
            {"type": "move_clip", "payload": {"targetClipId": "src_video__b", "toIn": 0.0}}],
         "样例_片段重排.mp4", 1.0),
        ("淡入", [{"type": "effect", "payload": {"targetClipId": "src_video",
                                                 "effectType": "fade_in",
                                                 "params": {"duration": 2.0}}}],
         "样例_淡入效果.mp4", 0.1),
    ]

    rows = []
    for name, ops, out_name, probe_t in cases:
        tl, _logs, warns = apply_operations(base_tl, ops, strict=True)
        assert warns == [], f"{name} 出现警告: {warns}"
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        p, msg = render(tl, base_dir=str(d))
        if not p:
            print(f"[FAIL] {name}: {msg}")
            rows.append({"name": name, "ok": False, "err": msg})
            continue
        shutil.copy2(p, SAMPLES / out_name)

        px_new, px_old = pixel(p, probe_t), pixel(p_base, probe_t)
        rms_new, rms_old = rms(pcm(p)), rms(pcm(p_base))
        changed = (px_new != px_old) or (abs(rms_new - rms_old) > 50)
        rows.append({"name": name, "sample": out_name,
                     "dur": round(dur(p), 2), "px": px_new, "px_base": px_old,
                     "rms": round(rms_new, 1), "rms_base": round(rms_old, 1),
                     "ok": changed})
        print(f"[{'OK ' if changed else 'FAIL'}] {name}")
        print(f"       {out_name}  时长={dur(p):.2f}s")
        print(f"       t={probe_t}s 像素 基线{px_old} -> 之后{px_new}")
        print(f"       音频 RMS 基线{rms_old:.1f} -> 之后{rms_new:.1f}")

    ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n=== 真实素材验收：{ok}/{len(rows)} ===")

    # 黑白效果：多点扫描彩度（max|R-G|,|G-B|,|R-B|），比单帧更有说服力
    gray_sample = SAMPLES / "样例_黑白效果.mp4"
    if gray_sample.exists():
        times = [0.5, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
        def chroma(px):
            return max(abs(px[0] - px[1]), abs(px[1] - px[2]), abs(px[0] - px[2]))
        before = [chroma(pixel(p_base, t)) for t in times]
        after = [chroma(pixel(str(gray_sample), t)) for t in times]
        print("\n[黑白效果·彩度扫描] t(s): " + " ".join(f"{t:>5.1f}" for t in times))
        print("              原片彩度: " + " ".join(f"{v:>5d}" for v in before))
        print("              黑白后彩度: " + " ".join(f"{v:>5d}" for v in after))
        print(f"              原片最大彩度={max(before)} -> 黑白后最大彩度={max(after)}"
              f"  {'✅ 全帧去色' if max(after) <= 3 else '❌ 仍有彩色'}")
        rows.append({"name": "黑白·彩度扫描", "max_chroma_before": max(before),
                     "max_chroma_after": max(after), "ok": max(after) <= 3})

    (OUT / "after_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
