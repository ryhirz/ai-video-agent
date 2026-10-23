"""Webhook 终态回调（全栈层第四步，契约冻结于 docs/全栈层/04-API §7）。

- 触发时机：任务进入终态（success / failed / cancelled / timeout）后回调
- 方向：服务端 POST 用户提交的 webhook_url
- 请求体：与 GET /render/{id} 响应同构 + event + finished_at（见 build_payload）
- 重试：非 2xx 视为失败，最多重试 3 次，间隔 10s/60s/300s（指数退避简化版）
- 超时：单次回调 10s
- 安全：本版不加密签（无认证体系的一部分），调用方应自行校验 task_id

_RETRIES 设计为模块级常量：测试可直接 patch 为 (0.0, 0.0, 0.0) 加速；
生产语义不变。回调失败只影响本次投递，不改变任务终态、不抛出异常。
"""
from __future__ import annotations

import json
import time as _time
import urllib.request
from typing import Any, Dict

RETRIES: tuple = (10.0, 60.0, 300.0)   # 3 次重试间隔（秒），测试可 patch
SINGLE_TIMEOUT = 10.0                  # 单次回调超时（秒），契约冻结值


def build_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    """按冻结契约构造回调体：与状态查询响应同构 + event + finished_at。"""
    return {
        "event": "task.finished",
        "task_id": row["id"],
        "state": row["state"],
        "progress": row["progress"],
        "result_url": row["result_url"],
        "error_msg": row["error_msg"],
        "finished_at": row["finished_at"],
    }


def fire_webhook(url: str, payload: Dict[str, Any],
                 retries: tuple | None = None) -> bool:
    """POST 回调到 url；非 2xx / 网络异常按重试间隔重试。

    retries=None（默认）时运行时读取模块级 RETRIES —— 这样测试 patch
    wh.RETRIES 才能生效（默认参数在定义时绑定，patch 不改变已绑定值）。
    返回 True = 某次尝试拿到 2xx；False = 全部尝试耗尽。
    所有异常在此吞掉：webhook 属尽力投递，绝不能反过来把任务体炸成 FAILED。
    """
    if retries is None:
        retries = RETRIES
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(len(retries) + 1):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=SINGLE_TIMEOUT) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:                      # noqa: BLE001 —— 尽力投递
            pass
        if attempt < len(retries):
            _time.sleep(retries[attempt])
    return False
