"""真模型 → 真 App 路径 → 真出片：把「用户最终会拿到什么」摆出来。

与另外两个验证的分工：
    `_verify/verify_llm_real.py`        真模型 + provider/planner 层（看模型输出与事实核对）
    `tests/test_ui.py::TestSemanticDeleteGuard`  stub 回放 + do_plan 应用层（看 UI 是否会提示）
    **本脚本**                            **真模型 + do_plan + do_apply_compile + 真渲染 + ffprobe**
    —— 三者的交集：真实模型下，用户点完③⑤⑥到底拿到一段什么样的成片。

用法（需先起服务 E:\\AI_models\\Qwen_Model\\start_qwen.bat）：
    .venv\\Scripts\\python.exe _verify\\verify_ui_llm_real.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.agent import RuleBasedPlanner  # noqa: E402
from ai_video_agent.footage_index import FootageIndex  # noqa: E402
from ai_video_agent.ingest import build_footage_index, build_initial_timeline  # noqa: E402
from ai_video_agent.ui import do_plan, do_apply_compile, APPLY_OUTPUT_KEYS  # noqa: E402

FFPROBE = ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe"
CLIP = ROOT / "测试素材2_多目标.mp4"
ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8001/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3-0.6b")
OUTDIR = ROOT / "_verify" / "ui_llm_out"


def probe(path: str) -> dict:
    def q(args):
        return subprocess.run([str(FFPROBE), "-v", "error", *args, path],
                              capture_output=True, text=True).stdout.strip()
    return {"duration": float(q(["-show_entries", "format=duration", "-of", "csv=p=0"]) or 0),
            "video": q(["-select_streams", "v:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0"]),
            "audio": q(["-select_streams", "a:0", "-show_entries", "stream=duration",
                        "-of", "csv=p=0"])}


def _index() -> FootageIndex:
    cache = ROOT / "_verify" / "_idx_material2.json"
    if cache.exists():
        return FootageIndex.loads(cache.read_text(encoding="utf-8"))
    from ai_video_agent.understand import detect_objects, transcribe
    idx = build_footage_index(str(CLIP))
    idx.visual_segments = detect_objects(idx)
    idx.transcript_segments = transcribe(idx)
    cache.write_text(idx.dumps(), encoding="utf-8")
    return idx


def run(idx: FootageIndex, nl: str, mode: str) -> dict:
    """走**应用层**处理器：do_plan -> do_apply_compile -> 真 ffprobe。"""
    tl = build_initial_timeline(idx)
    ops, _ops_json, status = do_plan(
        nl, tl, idx, mode, "", "", "", ENDPOINT, MODEL) if mode == "local" else do_plan(
        nl, tl, idx, mode, "", "", "", "", "")
    if not ops:
        return {"mode": mode, "nl": nl, "status": status, "ops": [], "error": "未产出操作"}
    res = dict(zip(APPLY_OUTPUT_KEYS, do_apply_compile(tl, ops)))
    out = res.get("video_out") or ""
    info = probe(out) if out and Path(out).exists() else {}
    return {"mode": mode, "nl": nl, "status": status, "ops": ops,
            "apply_status": res.get("apply_status", ""),
            "output": out, "probe": info,
            "ranges": [(o.get("payload", {}).get("start"), o.get("payload", {}).get("end"))
                       for o in ops if o.get("type") == "delete_range"]}


def main() -> None:
    idx = _index()
    print(f"素材：{CLIP.name}  时长 {idx.metadata.get('duration')}s"
          f"；镜头 {len(idx.scenes or [])} / 视觉段 {len(idx.visual_segments)}"
          f" / 台词 {len(idx.transcript_segments)}")
    NL = "删掉有公交车的片段"
    report = []
    for mode, label in (("local", "真模型 Qwen3-0.6B（走 JsonPlanner）"),
                        ("rule", "规则 Planner（离线，确定性）")):
        print("\n" + "=" * 72)
        print(f"【{label}】指令：{NL}")
        r = run(idx, NL, mode)
        print(f"  规划出的删除区间：{r.get('ranges')}")
        print(f"  ③ 状态：\n    {r.get('status','')!r}")
        if r.get("apply_status"):
            print(f"  ④ 状态：{r['apply_status'][:200]}")
        if r.get("probe"):
            print(f"  成片：{Path(r['output']).name} → "
                  f"{r['probe']['duration']:.2f}s "
                  f"(video={r['probe']['video']}, audio={r['probe']['audio']})")
        else:
            print("  成片：未产出")
        report.append(r)

    lit = report[0]
    analytic = report[1]
    print("\n" + "=" * 72)
    print("对比（这就是留给你拍板的那个取舍）：")
    if lit.get("probe") and analytic.get("probe"):
        print(f"  真模型成片 {lit['probe']['duration']:.2f}s   vs   "
              f"规则版 {analytic['probe']['duration']:.2f}s")
        diff = abs(lit["probe"]["duration"] - analytic["probe"]["duration"])
        print(f"  差 {diff:.2f}s —— {'一致 ✅' if diff < 0.05 else '不一致 ❌（模型答错了）'}")
    guarded = "删除区间与素材事实不一致" in (lit.get("status") or "")
    print(f"  事实核对闸是否给出提示：{'是 ✅' if guarded else '否 ❌'}")

    (ROOT / "_verify" / "ui_llm_real.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("明细已写入 _verify/ui_llm_real.json")


if __name__ == "__main__":
    main()
