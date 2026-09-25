"""真机端到端：用**真实模型**跑一遍 LLM 规划路径（补上"LLM 路径只被 stub 验证过"这个缺口）。

为什么需要这个脚本：
    单元测试里的 stub 只校验"请求发出去了"（model 名 / messages 非空 / 鉴权头），
    模型返回什么由 stub 自己造 —— 于是"模型看到新提示词后到底会不会照做"从未被验证。
    本脚本连**真实端点**（默认本地 Qwen3-0.6B，OpenAI 兼容）跑完整链路：
        提示词拼装 → 真实生成 → _extract_json_array → _validate_ops → apply_operations

重点观察（这些才是 stub 永远测不出来的）：
    1. 模型有没有**照规则 3b** 用「镜头 X-Y」区间，而不是台词自己的小时间戳（8.15-9.34s）；
    2. 输出需不需要走**容错提取 / 重试**才解析成功（弱模型上很常见）；
    3. 解析失败时抛出的错误信息是否**可读**（含原始输出片段 + 可用 id）；
    4. 把模型输出**真的应用**到时间轴上，看是否删干净（保留区间与目标镜头无交集）。

用法：
    # 先起服务：E:\\AI_models\\Qwen_Model\\start_qwen.bat （或 deploy venv 跑 serve_qwen3.py）
    .venv\\Scripts\\python.exe _verify\\verify_llm_real.py
    # 换端点/模型（如云端有 Key 时）：
    LLM_ENDPOINT=http://127.0.0.1:8001/v1 LLM_MODEL=qwen3-0.6b .venv\\Scripts\\python.exe _verify\\verify_llm_real.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_video_agent.agent import (EditingAgent, JsonPlanner, facts_block,  # noqa: E402
                                  semantic_delete_discrepancies)
from ai_video_agent.footage_index import FootageIndex  # noqa: E402
from ai_video_agent.ingest import build_footage_index, build_initial_timeline  # noqa: E402
from ai_video_agent.operations import apply_operations  # noqa: E402
from ai_video_agent.ui import OpenAICompatibleProvider  # noqa: E402

CLIP = ROOT / "测试素材2_多目标.mp4"
ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8001/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen3-0.6b")

# 素材2 上的期望答案：3 个公交镜头（0–4 / 8–12 / 16–20s）
EXPECT_BUS_SHOTS = [(0.0, 4.0), (8.0, 12.0), (16.0, 20.0)]

CASES = [
    ("删掉说公交车的片段", "按台词删除 —— 必须用「镜头 X-Y」聚合区间，不是台词小时间戳"),
    ("删掉有公交车的片段", "按视觉标签删除 —— 必须给 3 个区间各一条 delete_range"),
    ("去掉前2秒", "简单 trim —— 基线，弱模型也该答对"),
]


def _probe_endpoint() -> bool:
    from urllib import request as ureq
    try:
        with ureq.urlopen(ENDPOINT.rstrip("/").rsplit("/v1", 1)[0] + "/health",
                          timeout=3) as r:
            print(f"端点探活 OK：{r.read().decode('utf-8', 'ignore')}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ 端点探活失败（{e}）；仍会尝试 /v1 请求。")
        return False


def _build_or_load_index() -> FootageIndex:
    """真实 M2 摄取（约 30–40s）。结果缓存成 json，便于复跑。"""
    cache = ROOT / "_verify" / "_idx_material2.json"
    if cache.exists():
        print(f"复用缓存 index：{cache.name}")
        return FootageIndex.loads(cache.read_text(encoding="utf-8"))
    from ai_video_agent.understand import detect_objects, transcribe
    idx = build_footage_index(str(CLIP))
    print("M2 视觉理解（YOLO + VAD 分段转写）…")
    idx.visual_segments = detect_objects(idx)
    idx.transcript_segments = transcribe(idx)
    cache.write_text(idx.dumps(), encoding="utf-8")
    print(f"  索引已缓存：{cache.name}")
    return idx


def main() -> None:
    _probe_endpoint()
    idx = _build_or_load_index()
    tl0 = build_initial_timeline(idx)

    # 提示词自检：确认"按镜头聚合"那行确实进了提示词（与单测呼应，但这里是真实拼装）
    planner = JsonPlanner(OpenAICompatibleProvider(ENDPOINT, API_KEY, MODEL))
    prompt = planner.build_prompt("删掉说公交车的片段", tl0, idx)
    facts_sec = prompt[prompt.index("【素材理解事实（M2）】"):prompt.index("【用户指令】")]
    print(f"\n提示词长度 {len(prompt)} 字符；事实段含「按镜头聚合」={ '按镜头聚合' in facts_sec }")
    print(f"事实段含「镜头 8.00-12.00s」={ '镜头 8.00-12.00s' in facts_sec }")
    print(f"事实段不含半句小时间戳 8.15-9.34s = { '8.15-9.34s' not in facts_sec }")

    provider = OpenAICompatibleProvider(ENDPOINT, API_KEY, MODEL)
    agent = EditingAgent(provider=provider)

    report = []
    for nl, why in CASES:
        tl = build_initial_timeline(idx)
        print("\n" + "=" * 70)
        print(f"指令：{nl}\n意图：{why}")
        t0 = time.time()
        raw = ""
        err = ""
        ops = []
        try:
            raw = provider.complete(planner.build_prompt(nl, tl, idx))
            print(f"  原始输出（{len(raw)} 字符，{time.time()-t0:.1f}s）：{raw[:400]!r}")
            try:
                ops = agent.plan(nl, tl, idx)      # 内部还会再调一次（含重试）
            except ValueError as e:
                err = str(e)
                print(f"  ❌ 解析/校验失败：{err[:300]}")
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            print(f"  ❌ 调用失败：{err[:300]}")

        ranges, kept, warns, applied = [], [], [], False
        if ops:
            ranges = [(o.get("payload", {}).get("start"), o.get("payload", {}).get("end"))
                      for o in ops if o.get("type") == "delete_range"]
            print(f"  解析出 {len(ops)} 条操作：{ops}")
            try:
                tl2, _logs, warns = apply_operations(tl, ops, strict=False)
                kept = sorted((c.sourceIn, c.sourceOut)
                              for c in tl2.clips if c.track == "v_main")
                applied = True
                print(f"  应用后保留片段：{kept}")
                if warns:
                    print(f"  ⚠️ 跳过 {len(warns)} 条：{warns}")
            except Exception as e:  # noqa: BLE001
                print(f"  ❌ 应用失败：{e}")

        # 判定：删除区间是否命中期望的三个公交镜头
        hit = sorted(r for r in ranges if r in [tuple(x) for x in EXPECT_BUS_SHOTS])
        expect_speech = nl.startswith("删掉说")
        # 新加的第 5 项：**用 M2 事实核对模型输出**（真机缺口催生的那一道闸）
        discrepancy = semantic_delete_discrepancies(nl, idx, ops)
        caught = bool(discrepancy)
        if discrepancy:
            print(f"  🛡️ 事实核对拦下 {len(discrepancy)} 处不一致：")
            for d in discrepancy:
                print(f"       - {d}")
        else:
            print("  🛡️ 事实核对：与 M2 事实一致（或无语义期望值）")
        verdict = "—"
        if expect_speech:
            if not ops:
                verdict = "❌ 未产出操作"
            elif sorted(tuple(r) for r in ranges) == [tuple(x) for x in EXPECT_BUS_SHOTS]:
                verdict = "✅ 完全正确（3 个镜头区间）"
            elif hit:
                verdict = f"⚠️ 部分正确（命中 {len(hit)}/3 个镜头）"
            else:
                used_small = any(r for r in ranges
                                 if r[0] is not None and 8.0 <= float(r[0]) <= 10.0
                                 and float(r[1]) - float(r[0]) < 3)
                verdict = ("❌ 用了台词小时间戳（正是规则 3b 禁止的）" if used_small
                           else "❌ 区间不对")
        if discrepancy:
            verdict += f"；🛡️ 事实核对已拦下（{len(discrepancy)} 处不一致，会展示给用户）"
        elif expect_speech and ops:
            verdict += "；🛡️ 事实核对通过"
        print(f"  → 判定：{verdict}")
        report.append({"nl": nl, "raw_len": len(raw), "raw_head": raw[:300],
                       "ops": ops, "ranges": ranges, "kept": kept, "warns": warns,
                       "applied": applied, "error": err, "verdict": verdict,
                       "discrepancy": discrepancy, "guard_caught": caught,
                       "elapsed_s": round(time.time() - t0, 1)})

    out = ROOT / "_verify" / "llm_real.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
