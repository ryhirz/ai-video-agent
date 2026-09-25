# M4 编辑智能体 · 验证报告

**日期**：2026-09-18  ·  **里程碑**：M1→M2→M3→M4 整链闭环（NL → 结构化操作 → 确定性编译）

## 实现决策（合理默认，已文档化假设）

用户未逐条回复 M4 的两个开放问题，按其一贯习惯走**合理默认 + 文档化假设**推进：

1. **不引入 LangGraph 重依赖** → 轻量自定义 `EditingAgent`（本沙箱装重包网络成本高，且轻量版现在即可离线验证整链）。
2. **LLM 后端推迟接入** → 默认 `RuleBasedPlanner`（确定性、可单测、无 Key 可用）；真实 LLM（DeepSeek/OpenAI/本地 Qwen）通过 `LLMProvider` 协议热插拔，由 `JsonPlanner` 消费。接入时只需实现 `LLMProvider.complete()` 并传入 `EditingAgent(provider=...)`。

## 交付物

`ai_video_agent/agent.py`：

| 组件 | 职责 |
|------|------|
| `EditingAgent` | 门面：`plan()` 出操作列表；`execute()` 规划并逐步 `apply_operation` |
| `LLMProvider`（Protocol） | 真 LLM 热插拔点：`complete(prompt) -> str` |
| `RuleBasedPlanner` | 离线确定性：cut / trim(头/尾) / add_subtitle / delete_clip / 语义检索 |
| `JsonPlanner` | 真 LLM 路径：把操作 schema 进提示词，解析模型 JSON 输出 |
| `_ZH2EN` | 中文物体词→COCO 英文标签（公交→bus、人→person…可扩充） |

## 测试结果

`tests/test_agent.py` **7 passed**；全量 **23 passed（20.25s）**。

**核心验证——语义检索闭环（M2 理解 × M4 规划 × M3 编译）**：
- 输入 FootageIndex 含 `bus` 视觉段 `(3.0, 5.0)`，指令 `"删掉有bus的片段"`
- → `RuleBasedPlanner` 产出 `[cut@3, cut@5(on __b), delete __b__a]`
- → 逐步 `apply_operation` 后 Timeline 剩 2 片 `(0,3)` 与 `(5,10)`，`(3,5)` 区间被剔除
- 证明"自然语言检索素材 → 结构化操作 → 确定性 FFmpeg 编译"整链跑通

## 踩坑记录

- cut 正则初版漏了"在第2秒"里的"第"字（在/于 后应有 `第?`），导致"在第2秒切"匹配失败；已修。

## 待解 / 下一步

1. **M4 真 LLM 后端**：待用户拍板（云端 Key vs 本地 Qwen）后实现 `LLMProvider` 接入。
2. **M2 FunASR 真实转写**：仍待带音轨样本 + 拍板 Paraformer 模型参数（当前无音轨测试片不触发下载）。
3. **M5（Gradio UI）**：按锁定计划接续，做最小可用演示界面。
