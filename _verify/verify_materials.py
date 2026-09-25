"""验证第二批测试素材（`测试素材2_多目标.mp4` / `测试素材3_竖屏音乐.mp4`）是否真的可用。

不看代码猜，用硬测量逐项钉死：
  1. 规格      —— ffprobe：时长/分辨率/帧率/音轨
  2. 场景切换  —— select='gt(scene,0.25)' 切换点个数（应等于"幕数 - 1"）
  3. 静音结构  —— silencedetect（素材2 应有规律语音/静音交替；素材3 应全程有声无静音）
  4. 画面运动  —— 逐帧小尺寸灰度差分：素材2 幕内应 ≈0（静态图），素材3 幕内应 >0（推镜真在动）
  5. YOLO      —— 抽帧跑 yolov8n，确认能检出目标（素材是否"有内容可言"）
  6. 端到端    —— 素材2 跑「删掉有bus的片段」全链路，应删掉 3 个 bus 区间、成片 ≈12s

产物写到 `_verify/mat_out/`（已 gitignore）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.agent import RuleBasedPlanner           # noqa: E402
from ai_video_agent.ingest import (                          # noqa: E402
    build_footage_index, build_initial_timeline, scene_cuts as ingest_scene_cuts)
from ai_video_agent.operations import apply_operations      # noqa: E402
from ai_video_agent.render import render                    # noqa: E402
from ai_video_agent.understand import detect_objects        # noqa: E402

FFMPEG = ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
OUT = ROOT / "_verify" / "mat_out"
OUT.mkdir(parents=True, exist_ok=True)

CLIPS = {
    "素材2_多目标": (ROOT / "测试素材2_多目标.mp4", 24.0, 6, 1280, 720),
    "素材3_竖屏音乐": (ROOT / "测试素材3_竖屏音乐.mp4", 20.0, 5, 720, 1280),
}


def _run(args, **kw):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", **kw)


def spec(path: Path) -> dict:
    r = _run([FFPROBE, "-v", "error", "-show_entries",
              "format=duration", "-show_entries",
              "stream=codec_type,width,height,r_frame_rate,channels,sample_rate",
              "-of", "json", path])
    return json.loads(r.stdout or "{}")


def scene_cuts(path: Path) -> list[float]:
    """复用项目自己的镜头检测（OpenCV 1Hz 直方图，阈 0.4）。

    注意：不要在这个脚本里另写一套 ffmpeg `select='gt(scene,..)'` 版本 ——
    实测那套在带淡入淡出的素材上会得出 0 个切点（与项目实现结论完全相反），
    差点把"验证脚本的实现 bug"误判成"素材没内容"。
    """
    return ingest_scene_cuts(str(path))


def silences(path: Path) -> list[tuple[float, float]]:
    r = _run([FFMPEG, "-hide_banner", "-i", path, "-af",
              "silencedetect=n=-35dB:d=0.15", "-f", "null", "-"])
    starts, out = [], []
    for line in r.stderr.splitlines():
        if "silence_start:" in line:
            starts.append(round(float(line.split("silence_start:")[1].split()[0]), 3))
        elif "silence_end:" in line:
            e = round(float(line.split("silence_end:")[1].split()[0]), 3)
            out.append((starts.pop(0) if starts else -1.0, e))
    if starts:  # 收尾静音（到片尾）
        r2 = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", path])
        out.append((starts[0], round(float(r2.stdout.strip() or 0), 3)))
    return out


def frame_diffs(path: Path, w: int, h: int) -> np.ndarray:
    """逐帧灰度差分序列：值大=画面在变，≈0=画面静止。"""
    r = subprocess.run(
        [str(FFMPEG), "-v", "error", "-i", str(path), "-vf",
         f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True)
    buf = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(buf) // (w * h)
    if n < 2:
        return np.zeros(0)
    fr = buf[: n * w * h].reshape(n, w * h).astype(np.float32)
    return np.abs(np.diff(fr, axis=0)).mean(axis=1)


def yolo_labels(path: Path, times: list[float]) -> list[dict]:
    from ultralytics import YOLO
    frames = []
    for t in times:
        p = OUT / f"{path.stem}_t{t}.png"
        _run([FFMPEG, "-v", "error", "-ss", str(t), "-i", path,
              "-frames:v", "1", "-y", p])
        if p.exists():
            frames.append((t, p))
    model = YOLO(str(ROOT / "weights" / "yolov8n.pt"))
    rows = []
    for t, p in frames:
        res = model.predict(str(p), verbose=False, conf=0.25)
        names = []
        for b in res[0].boxes:
            names.append(model.names[int(b.cls[0])])
        rows.append({"t": t, "labels": sorted(set(names))})
    return rows


def main() -> None:
    report: dict = {}

    for name, (path, exp_dur, exp_acts, w, h) in CLIPS.items():
        print(f"\n{'=' * 62}\n== {name}  ({path.name})\n{'=' * 62}")
        if not path.exists():
            print("  ❌ 文件不存在")
            continue

        sp = spec(path)
        v = next((s for s in sp["streams"] if s["codec_type"] == "video"), {})
        a = next((s for s in sp["streams"] if s["codec_type"] == "audio"), None)
        dur = float(sp["format"]["duration"])
        print(f"  规格     时长={dur:.3f}s  {v.get('width')}x{v.get('height')}  "
              f"{v.get('r_frame_rate')}  音轨={'有' if a else '无'}"
              + (f" {a.get('sample_rate')}Hz/{a.get('channels')}ch" if a else ""))

        cuts = scene_cuts(path)
        print(f"  场景切换 {len(cuts)} 个 → {cuts}")

        sils = silences(path)
        print(f"  静音段   {len(sils)} 段 → {sils[:8]}{' ...' if len(sils) > 8 else ''}")

        d = frame_diffs(path, 96, 54)
        if d.size:
            # 幕内（避开前后 0.3s 的淡入淡出）与切换点附近分别看
            inside = []
            for k in range(exp_acts):
                lo, hi = int((k * 4.0 + 0.6) * 25), int(((k + 1) * 4.0 - 0.6) * 25)
                seg = d[lo:hi]
                if seg.size:
                    inside.append(float(seg.mean()))
            print(f"  画面运动 各幕内平均帧差 {[round(x, 2) for x in inside]}  "
                  f"整片最大 {d.max():.2f}")

        labels = yolo_labels(path, [t for t in np.arange(1.0, exp_dur, 2.0)])
        seen = {}
        for r in labels:
            for lb in r["labels"]:
                seen.setdefault(lb, 0)
                seen[lb] += 1
        print(f"  YOLO     检出类别（按出现帧数）: "
              f"{sorted(seen.items(), key=lambda kv: -kv[1])}")

        report[name] = {
            "file": path.name, "duration": dur,
            "resolution": [v.get("width"), v.get("height")],
            "fps": v.get("r_frame_rate"), "has_audio": bool(a),
            "scene_cuts": cuts, "silences": sils,
            "yolo_labels": seen, "yolo_frames": labels,
        }

    # ---------------- 端到端：素材2 跑真实指令 ----------------
    print(f"\n{'=' * 62}\n== 端到端：素材2 + 「删掉有bus的片段」\n{'=' * 62}")
    clip2 = ROOT / "测试素材2_多目标.mp4"
    idx = build_footage_index(str(clip2))
    # 注意 detect_objects 返回的是 VisualSegment **列表**（不是原地改 index），
    # 直接当 index 用会 AttributeError —— 我第一次就踩了。
    idx.visual_segments = detect_objects(idx)
    tl = build_initial_timeline(idx)
    lbls: dict = {}
    for s in idx.visual_segments:          # VisualSegment 的字段是 label/start/end
        lbls[s.label] = lbls.get(s.label, 0) + 1
    print(f"  M2 视觉索引: 场景 {len(idx.scenes)} / 关键帧 {len(idx.keyframes)} / "
          f"视觉段 {len(idx.visual_segments)} / 类别 {sorted(lbls.items(), key=lambda kv: -kv[1])}")
    ops = RuleBasedPlanner().plan("删掉有bus的片段", tl, idx)
    print(f"  规划操作 {len(ops)} 条:")
    for o in ops:
        print(f"    {o['type']} {o.get('payload')}")

    tl2, logs, warns = apply_operations(tl, ops, strict=True)
    v_keep = sorted((c.sourceIn, c.sourceOut) for c in tl2.clips if c.track == "v_main")
    a_keep = sorted((c.sourceIn, c.sourceOut) for c in tl2.clips if c.track == "a_main")
    print(f"  保留区间 视频={[(round(x, 2), round(y, 2)) for x, y in v_keep]}")
    print(f"          音频={[(round(x, 2), round(y, 2)) for x, y in a_keep]}")
    print(f"  A/V 镜像: {[(round(x, 2), round(y, 2)) for x, y in v_keep] == [(round(x, 2), round(y, 2)) for x, y in a_keep]}")
    print(f"  警告 {len(warns)} 条: {warns}")

    out_path, msg = render(tl2, base_dir=str(OUT))
    if out_path:
        r = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                  "-of", "csv=p=0", out_path])
        got = round(float(r.stdout.strip() or 0), 3)
        keep = round(sum(y - x for x, y in v_keep), 3)
        print(f"  ✅ 出片 {out_path}")
        print(f"     成片时长={got:.3f}s  期望={keep:.3f}s  "
              f"（源 {report.get('素材2_多目标', {}).get('duration', 0):.1f}s → "
              f"删掉 3 个 bus 区间）")
        report["e2e_clip2"] = {"ops": ops, "warns": warns,
                               "video_keep": v_keep, "audio_keep": a_keep,
                               "rendered": out_path, "duration": got,
                               "expected": keep, "logs": logs}
    else:
        print(f"  ❌ 渲染失败：{msg}")
        report["e2e_clip2"] = {"error": msg}

    (OUT / "materials_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {OUT / 'materials_report.json'}")


if __name__ == "__main__":
    main()
