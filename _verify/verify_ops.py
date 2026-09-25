"""9 类操作真实出片矩阵验证。

对契约里的 9 类操作各构造一条指令，走完整链路：
    build_footage_index -> build_initial_timeline -> apply_operations -> compile -> render -> ffprobe

判定标准不是"没报错"，而是**成片真的变了**（时长/流数/字幕/片段数符合预期）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ingest import build_footage_index, build_initial_timeline
from ai_video_agent.operations import apply_operations
from ai_video_agent.render import render
from ai_video_agent.timeline import Timeline
from ai_video_agent.compiler import compile as compile_tl

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIP = ROOT / "_verify" / "clip6.mp4"
OUT = ROOT / "_verify" / "ops_out"
OUT.mkdir(parents=True, exist_ok=True)


def probe_dur(path: str) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return -1.0


def probe_streams(path: str) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "stream=codec_type,duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    out = {"video": 0, "audio": 0}
    for line in r.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 1 and parts[0] in out:
            out[parts[0]] += 1
    return out


def base_tl() -> Timeline:
    idx = build_footage_index(str(CLIP))
    return build_initial_timeline(idx)


def run_case(name: str, ops, *, expect: str, check) -> dict:
    """跑一条用例，返回结果字典。"""
    tl0 = base_tl()
    res = {"op": name, "expect": expect}
    try:
        tl1, logs, warns = apply_operations(tl0, ops, strict=True)
        res["logs"] = logs
        res["warns"] = warns
        res["clips"] = [(c.id, c.track, round(c.sourceIn, 2), round(c.sourceOut, 2))
                        for c in tl1.clips]
        res["subs"] = len(tl1.subtitles)
        res["trans"] = len(tl1.transitions)
        res["effects"] = [len(c.effects) for c in tl1.clips]
        res["compiled"] = compile_tl(tl1)
    except Exception as e:  # noqa: BLE001
        res["apply_error"] = f"{type(e).__name__}: {e}"
        res["ok"] = False
        return res

    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path, msg = render(tl1, base_dir=str(out_dir))
    res["render_msg"] = msg if not path else "ok"
    if not path:
        res["ok"] = False
        return res

    res["dur"] = probe_dur(path)
    res["streams"] = probe_streams(path)
    res["path"] = path
    verdict, detail = check(tl1, res)
    res["ok"] = verdict
    res["detail"] = detail
    return res


CASES = [
    # ---- 1. cut ----
    dict(
        name="cut",
        ops=[{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}}],
        expect="视频被切成 2 段，成片时长不变(6s)，音视频同步切",
        check=lambda tl, r: (
            len([c for c in tl.clips if c.track == "v_main"]) == 2
            and len([c for c in tl.clips if c.track == "a_main"]) == 2
            and abs(r["dur"] - 6.0) < 0.15,
            f"v段={len([c for c in tl.clips if c.track=='v_main'])} "
            f"a段={len([c for c in tl.clips if c.track=='a_main'])} 时长={r['dur']:.2f}",
        ),
    ),
    # ---- 2. trim ----
    dict(
        name="trim",
        ops=[{"type": "trim", "payload": {"targetClipId": "src_video",
                                           "in": 1.0, "out": 5.0}}],
        expect="成片 4s，音频同步跟随",
        check=lambda tl, r: (
            abs(r["dur"] - 4.0) < 0.15 and r["streams"]["audio"] == 1,
            f"时长={r['dur']:.2f} 音频流={r['streams']['audio']}",
        ),
    ),
    # ---- 3. add_subtitle ----
    dict(
        name="add_subtitle",
        ops=[{"type": "add_subtitle", "payload": {"id": "s1", "text": "开场",
                                                   "start": 0.5, "end": 3.0}}],
        expect="得 1 条字幕，成片仍 6s，subs.ass 落盘",
        check=lambda tl, r: (
            len(tl.subtitles) == 1 and abs(r["dur"] - 6.0) < 0.15
            and (Path(r["path"]).parent / "subs.ass").exists(),
            f"字幕={len(tl.subtitles)} 时长={r['dur']:.2f} "
            f"ass={(Path(r['path']).parent / 'subs.ass').exists()}",
        ),
    ),
    # ---- 4. add_bgm ----
    dict(
        name="add_bgm",
        ops=[{"type": "add_bgm", "payload": {"sourceRef": str(CLIP),
                                             "in": 0.0, "out": 6.0,
                                             "volume": 0.3}}],
        expect="BGM 与主音频**混音**叠加（amix），成片仍 6s",
        check=lambda tl, r: (
            "amix" in r["compiled"]["audio_filter"] and abs(r["dur"] - 6.0) < 0.15,
            f"时长={r['dur']:.2f} 音频链含 amix={'amix' in r['compiled']['audio_filter']}",
        ),
    ),
    # ---- 5. delete_clip ----
    dict(
        name="delete_clip",
        ops=[{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
             {"type": "delete_clip", "payload": {"targetClipId": "src_video__b"}}],
        expect="删掉后半段 -> 成片 3s，且音频一并删（不留 6s 原声）",
        check=lambda tl, r: (
            abs(r["dur"] - 3.0) < 0.2
            and len([c for c in tl.clips if c.track == "a_main"]) == 1,
            f"时长={r['dur']:.2f} 剩余音频段={len([c for c in tl.clips if c.track=='a_main'])}",
        ),
    ),
    # ---- 6. delete_range ----
    dict(
        name="delete_range",
        ops=[{"type": "delete_range", "payload": {"start": 1.0, "end": 3.0}}],
        expect="成片 4s，视频/音频各 2 段且素材区间镜像",
        check=lambda tl, r: (
            abs(r["dur"] - 4.0) < 0.2
            and len([c for c in tl.clips if c.track == "v_main"]) == 2
            and len([c for c in tl.clips if c.track == "a_main"]) == 2
            and r["streams"]["video"] == 1 and r["streams"]["audio"] == 1,
            f"时长={r['dur']:.2f} "
            f"v段={len([c for c in tl.clips if c.track=='v_main'])} "
            f"a段={len([c for c in tl.clips if c.track=='a_main'])}",
        ),
    ),
    # ---- 7. move_clip ----
    dict(
        name="move_clip",
        ops=[{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
             {"type": "move_clip", "payload": {"targetClipId": "src_video__b",
                                               "toIn": 0.0}}],
        expect="两段视频顺序对调，成片仍 6s 但画面顺序变了",
        check=lambda tl, r: (
            abs(r["dur"] - 6.0) < 0.2
            and [c.id for c in sorted([c for c in tl.clips if c.track == "v_main"],
                                      key=lambda c: c.in_)] == ["src_video__b", "src_video__a"],
            "顺序=" + str([c.id for c in sorted(
                [c for c in tl.clips if c.track == "v_main"], key=lambda c: c.in_)]),
        ),
    ),
    # ---- 8. transition ----
    dict(
        name="transition",
        ops=[{"type": "cut", "payload": {"targetClipId": "src_video", "at": 3.0}},
             {"type": "transition", "payload": {"transitionType": "fade",
                                                "duration": 0.5,
                                                "clips": ["src_video__a", "src_video__b"]}}],
        expect="转场以 fade 滤镜进入成片（视频链含 fade=t=out / fade=t=in）",
        check=lambda tl, r: (
            "fade=t=out" in r["compiled"]["video_filter"]
            and "fade=t=in" in r["compiled"]["video_filter"]
            and abs(r["dur"] - 6.0) < 0.2,
            f"视频链含 fade_out={'fade=t=out' in r['compiled']['video_filter']} "
            f"fade_in={'fade=t=in' in r['compiled']['video_filter']} 时长={r['dur']:.2f}",
        ),
    ),
    # ---- 9. effect ----
    dict(
        name="effect",
        ops=[{"type": "effect", "payload": {"targetClipId": "src_video",
                                            "effectType": "grayscale",
                                            "params": {}}}],
        expect="灰度以 hue=s=0 进入成片（像素 R=G=B）",
        check=lambda tl, r: (
            "hue=s=0" in r["compiled"]["video_filter"] and abs(r["dur"] - 6.0) < 0.2,
            f"视频链含 hue=s=0={'hue=s=0' in r['compiled']['video_filter']} 时长={r['dur']:.2f}",
        ),
    ),
]


def ensure_fixture() -> None:
    """确保验证素材存在 —— 逻辑已提到共用夹具 `_verify/_fixture.py`。

    `verify_semantics.py` 也用同一份（它原本根本没生成素材，单独复跑会崩）。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _fixture import ensure_fixture as _ensure
    _ensure(verbose=True)


def main() -> None:
    ensure_fixture()
    print(f"验证素材: {CLIP}  时长={probe_dur(str(CLIP)):.2f}s  流={probe_streams(str(CLIP))}")
    print("=" * 100)
    rows = []
    for c in CASES:
        r = run_case(c["name"], c["ops"], expect=c["expect"], check=c["check"])
        rows.append(r)
        flag = "PASS" if r.get("ok") else "FAIL"
        print(f"[{flag}] {r['op']:<14s} 期望: {r['expect']}")
        if "apply_error" in r:
            print(f"         应用阶段报错: {r['apply_error']}")
        else:
            print(f"         实测: {r.get('detail', '')}")
            print(f"         成片: 时长={r.get('dur', -1):.2f}s 流={r.get('streams')} "
                  f"字幕={r.get('subs')} 转场={r.get('trans')} 效果={r.get('effects')}")
        print("-" * 100)

    ok = sum(1 for r in rows if r.get("ok"))
    print(f"\n通过 {ok}/{len(rows)}")
    (OUT / "matrix_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
