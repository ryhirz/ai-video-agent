"""新素材走 **UI 真实路径** 的端到端验收（①摄取 → ②M2 → ③生成操作 → ④应用并编译）。

补两个此前只是"设计意图"、并未实测的缺口：
  1. **素材3 是纯音乐、无语音** —— FunASR 拿到没有语音的音轨到底会怎样？
     是干净返回空（UI 优雅降级），还是吐出幻觉文本（会污染语义检索）？
  2. **新素材在界面上到底走不走得通** —— 之前只验了库函数路径，
     这里改用 `ai_video_agent.ui` 的模块级处理器（界面按钮按的就是这些），
     并且同时覆盖「横屏 24s」与「竖屏 20s」两种形态。

所有判定都用 ffprobe 实测产物，不看代码猜。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.ui import APPLY_OUTPUT_KEYS, do_apply_compile, do_ingest, do_plan, do_understand  # noqa: E402

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
OUT = ROOT / "_verify" / "newmat_out"
OUT.mkdir(parents=True, exist_ok=True)

CLIP2 = ROOT / "测试素材2_多目标.mp4"     # 24s 横屏 + 逐幕旁白
CLIP3 = ROOT / "测试素材3_竖屏音乐.mp4"   # 20s 竖屏 + 纯音乐（无语音）


def probe(path: str) -> dict:
    r = subprocess.run([str(FFPROBE), "-v", "error", "-show_entries",
                        "format=duration", "-show_entries",
                        "stream=codec_type,width,height", "-of", "json", path],
                       capture_output=True, text=True, encoding="utf-8")
    try:
        d = json.loads(r.stdout or "{}")
    except Exception:
        return {"duration": -1.0, "w": 0, "h": 0}
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration": round(float(d.get("format", {}).get("duration", 0) or 0), 3),
            "w": v.get("width", 0), "h": v.get("height", 0)}


def apply(tl, ops):
    return dict(zip(APPLY_OUTPUT_KEYS, do_apply_compile(tl, ops)))


def head(s: str, n: int = 220) -> str:
    return " / ".join(s.strip().splitlines()[:2])[:n]


def case(clip: Path, need_m2: bool, nl: str, want_dur: float, want_wh=None) -> dict:
    print(f"\n{'─' * 74}\n【{clip.name}】指令：{nl!r}\n{'─' * 74}")

    index, tl, meta, scenes, kf = do_ingest(str(clip))
    assert tl is not None, meta
    print(f"① 摄取  时长={index.metadata['duration']}s "
          f"{index.metadata.get('width')}x{index.metadata.get('height')} "
          f"镜头={len(index.scenes)} 关键帧={len(index.keyframes)}")
    print(f"   镜头边界 {scenes.replace(chr(10), '；')[:150]}")

    if need_m2:
        index, status2 = do_understand(index)
        print(f"② M2  视觉段 {len(index.visual_segments)} 条 / "
              f"转写段 {len(index.transcript_segments)} 条")
        for v in index.visual_segments:
            print(f"       [视觉] {v.label:<16s} {v.start:6.2f}s – {v.end:6.2f}s")
        for t in index.transcript_segments[:6]:
            print(f"       [转写] {t.start:6.2f}s – {t.end:6.2f}s  {t.text[:40]!r}")
        print(f"   ② 状态回显: {head(status2, 300)}")

    ops, ops_json, status3 = do_plan(nl, tl, index, "rule", "", "", "", "", "")
    assert ops, f"③ 未生成操作：{status3}"
    print(f"③ 规划 {len(ops)} 条: {json.dumps(ops, ensure_ascii=False)}")

    res = apply(tl, ops)
    v = res["video_out"]
    assert v, f"④ 未出片：{res['apply_status']}"
    p = probe(v)
    ok_dur = abs(p["duration"] - want_dur) < 0.25
    ok_wh = (want_wh is None) or ((p["w"], p["h"]) == want_wh)
    print(f"④ 应用 状态: {res['apply_status'].splitlines()[0]}")
    if "被跳过" in res["apply_status"]:
        print(f"   ⚠️ 警告段: {head(res['apply_status'].split('被跳过', 1)[1], 240)}")
    print(f"   成片 {v}")
    print(f"   实测 时长={p['duration']}s（期望 {want_dur}）  分辨率={p['w']}x{p['h']}"
          f"   -> {'PASS' if (ok_dur and ok_wh) else 'FAIL'}")
    return {"clip": clip.name, "nl": nl, "ops": ops, "dur": p["duration"],
            "want": want_dur, "wh": [p["w"], p["h"]], "ok": ok_dur and ok_wh,
            "status4": res["apply_status"].splitlines()[0], "video": v}


def main() -> None:
    rows = []

    # ---- 素材3 第一：先看"无语音音轨"喂给 ASR 会怎样（这是真正未知的一点）----
    print("=" * 74)
    print("缺口 1：纯音乐音轨（无语音）→ FunASR 的行为")
    print("=" * 74)
    idx3, tl3, meta3, _s, _k = do_ingest(str(CLIP3))
    idx3, status = do_understand(idx3)
    n_tr = len(idx3.transcript_segments)
    text = "".join(t.text for t in idx3.transcript_segments).strip()
    print(f"   转写段数 = {n_tr} / 文本 = {text[:120]!r}")
    if n_tr == 0 or not text:
        print("   ✅ 干净返回空 —— UI 会走「无语音」展示，不会污染语义检索")
    else:
        print("   ⚠️ 有输出：需人眼判断是真实语音还是音乐的幻觉文本（见下文逐条）")
    print(f"   ② 状态回显: {head(status, 300)}")
    rows.append({"case": "M2 on music-only audio", "n_transcript": n_tr,
                 "text": text[:200], "status": status.splitlines()[:3]})

    # ---- 素材2：多区间删除，走完整 UI 路径 ----
    rows.append(case(CLIP2, True, "删掉有bus的片段", 12.0, (1280, 720)))

    # ---- 素材3：竖屏 + 去头 + 字幕，走完整 UI 路径 ----
    rows.append(case(CLIP3, True, "去掉前2秒", 18.0, (720, 1280)))

    ok = sum(1 for r in rows[1:] if r.get("ok"))
    print(f"\n{'=' * 74}\n=== UI 路径验收（新素材）：通过 {ok}/{len(rows) - 1} ===")
    (OUT / "ui_newmat_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已写入 {OUT / 'ui_newmat_result.json'}")


if __name__ == "__main__":
    main()
