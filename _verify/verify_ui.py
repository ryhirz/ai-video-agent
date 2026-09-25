"""UI 层端到端验收：真实素材跑 ①摄取 -> ③生成操作 -> ④应用并编译（真实出片）。

全部走 ai_video_agent.ui 的模块级处理器（界面按的就是这些函数），
因此结论可直接代表界面上点按钮的结果。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from ai_video_agent.footage_index import FootageIndex  # noqa: E402
from ai_video_agent.ui import (APPLY_OUTPUT_KEYS, build_app,  # noqa: E402
                               do_apply_compile, do_ingest, do_plan)

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIP = str(ROOT / "测试素材.mp4")
M2_JSON = ROOT / "_verify" / "m2_result.json"


def dur(p: str) -> float:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return -1.0


def n_streams(p: str) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "stream=codec_type", "-of", "csv=p=0", p],
                       capture_output=True, text=True)
    out = {"video": 0, "audio": 0}
    for line in r.stdout.split():
        if line in out:
            out[line] += 1
    return out


def apply(tl, ops):
    res = do_apply_compile(tl, ops)
    return dict(zip(APPLY_OUTPUT_KEYS, res))


def main() -> None:
    # ---------- 0) 界面接线自省 ----------
    app = build_app()
    n_fns = len(app.fns)
    print(f"[接线] build_app() 成功，注册事件 {n_fns} 个，组件 {len(app.blocks)} 个")
    for bf in app.fns.values():
        if getattr(bf, "fn", None) is do_apply_compile:
            kinds = [type(c).__name__ for c in bf.outputs]
            print(f"[接线] ④ 输出组件类型 = {kinds}")
            print(f"[接线] 契约 APPLY_OUTPUT_KEYS = {list(APPLY_OUTPUT_KEYS)}")
            assert kinds == ["State", "Markdown", "Video", "Code", "Code", "Code"], kinds
            print("[接线] ✅ 组件顺序与契约逐位一致")
    print("=" * 78)

    # ---------- 1) ① 摄取 ----------
    index, tl, meta, scenes, kf = do_ingest(CLIP)
    assert tl is not None, meta
    print(f"[①摄取] 时长={index.metadata['duration']}s "
          f"分辨率={index.metadata.get('width')}x{index.metadata.get('height')} "
          f"场景={len(index.scenes)} 关键帧={len(index.keyframes)}")
    print(f"         clips={[c.id for c in tl.clips]} tracks={[t.id for t in tl.tracks]}")
    print("=" * 78)

    # ---------- 2) 可选：载入 M2 增强索引（复用后台跑的 M2 结果，避免重复加载模型） ----------
    index_m2 = None
    if M2_JSON.exists():
        d = json.loads(M2_JSON.read_text(encoding="utf-8"))
        if d.get("index_full"):
            index_m2 = FootageIndex.from_dict(d["index_full"])
            print(f"[M2] 已载入增强索引：视觉段 {len(index_m2.visual_segments)} 条 "
                  f"转写段 {len(index_m2.transcript_segments)} 条")
            for v in index_m2.visual_segments:
                print(f"        {v.label:<8s} {v.start:5.1f}s – {v.end:5.1f}s")
            print("=" * 78)

    # ---------- 3) 逐条指令跑 ③④ ----------
    cases = [
        ("去掉前2秒", "trim", 13.0, 1),
        ("去掉后3秒", "trim", 12.0, 1),
        ("在第2秒切一刀", "cut", 15.0, 1),
        ('加字幕"开场"', "add_subtitle", 15.0, 1),
        ("黑白", "effect", 15.0, 1),
        ("模糊", "effect", 15.0, 1),
    ]
    if index_m2 is not None:
        cases.append(("删掉有bus的片段", "delete_range", 8.0, 1))
        cases.append(('删掉有bus的片段，加字幕"开场"', "combo", 8.0, 1))

    rows = []
    for nl, kind, want_dur, want_audio in cases:
        idx_use = index_m2 if ("bus" in nl and index_m2 is not None) else index
        ops, ops_json, status3 = do_plan(nl, tl, idx_use, "rule", "", "", "", "", "")
        assert ops, f"{nl} -> 未生成操作：{status3}"
        res = apply(tl, ops)
        v = res["video_out"]
        got_dur = dur(v) if v else -1.0
        ok = bool(v) and abs(got_dur - want_dur) < 0.25
        rows.append(dict(nl=nl, ops=ops, want=want_dur, got=got_dur, ok=ok,
                         status=res["apply_status"].splitlines()[0],
                         video=v))
        print(f"[③④] 指令: {nl!r}")
        print(f"      操作: {json.dumps(ops, ensure_ascii=False)}")
        print(f"      状态: {res['apply_status'].splitlines()[0]}")
        print(f"      成片: {v}")
        print(f"      时长: 期望 {want_dur}s / 实测 {got_dur:.2f}s / 流={n_streams(v) if v else None}"
              f"  -> {'PASS' if ok else 'FAIL'}")
        if "被跳过" in res["apply_status"]:
            print(f"      ⚠️ 警告: {res['apply_status'].split('被跳过')[1][:200]}")
        print("-" * 78)

    ok_n = sum(1 for r in rows if r["ok"])
    print(f"\n=== UI 端到端：通过 {ok_n}/{len(rows)} ===")
    (ROOT / "_verify" / "ui_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
