"""模型客户端：OpenAI 兼容、可插拔（Qwen-VL / GLM / DeepSeek 文本）。

对应产品战略报告「模型端点选型」结论。本文件只定义契约与路由策略，
不在此发起网络请求——真实调用由 LangGraph Agent（M4）用 openai 库完成。

设计要点（回答"选哪家"）：模型做成可插拔的一层。
- 主模型：Qwen-VL（视觉+工具同框、DashScope 一端点多模型路由）
- 并列首选：GLM（Agent 基因强、永久免费 Flash 档，dev/CI 零成本）
- 降本节点：DeepSeek 文本档（纯文本规划/复盘步路由到这里）
- 兜底：Kimi / 豆包 作为可切换档
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    key: str
    name: str
    base_url: str
    api_key_env: str
    supports_vision: bool
    supports_tools: bool
    note: str = ""


# 候选端点（2026-09-17 检索；接入前以官方实时文档为准）
PROVIDERS: Dict[str, ModelConfig] = {
    "qwen-vl": ModelConfig(
        "qwen-vl", "qwen-vl-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY", True, True,
        "视觉+工具同框；DashScope 网关可代理 Qwen/DeepSeek/Kimi/GLM，天然多模型路由",
    ),
    "glm": ModelConfig(
        "glm", "glm-4.6v", "https://open.bigmodel.cn/api/paas/v4",
        "ZHIPU_API_KEY", True, True,
        "Agent/工具调用基因强；GLM-5.3-Flash 永久免费档，dev/CI 零成本",
    ),
    "deepseek": ModelConfig(
        "deepseek", "deepseek-chat", "https://api.deepseek.com/v1",
        "DEEPSEEK_API_KEY", False, True,
        "仅纯文本规划/复盘步降本；视觉实验档工具调用不稳，排除作主模型",
    ),
    "kimi": ModelConfig(
        "kimi", "kimi-k3", "https://api.moonshot.cn/v1",
        "MOONSHOT_API_KEY", True, True,
        "1M 上下文、工具编排强；长片难例兜底",
    ),
    "doubao": ModelConfig(
        "doubao", "seed-2.0-pro", "https://ark.cn-beijing.volces.com/api/v3",
        "ARK_API_KEY", True, True,
        "高并发兜底（10K QPS 口碑）",
    ),
}


def get_config(key: str) -> ModelConfig:
    if key not in PROVIDERS:
        raise KeyError(f"unknown model key: {key}; 可选: {list(PROVIDERS)}")
    return PROVIDERS[key]


def default_routing() -> Dict[str, str]:
    """返回 M4 Agent 的默认路由策略：主模型 + 文本步降本节点。"""
    return {"planner_vision": "qwen-vl", "planner_vision_alt": "glm",
            "text_planning": "deepseek", "hardcase": "kimi", "high_qps": "doubao"}
