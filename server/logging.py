"""统一日志配置模块（结构化日志 + 双模输出）。

开发环境（LOG_ENV=dev，默认）：彩色文本输出，便于本地调试。
生产环境（LOG_ENV=prod）：单行 JSON 输出，便于日志采集工具解析。

接入点（两处都是进程入口，worker 是独立进程必须各自初始化）：
- server/app.py 模块顶部（uvicorn server.app:app）
- server/tasks.py 模块顶部（celery -A server.tasks worker）

环境变量：
- LOG_ENV   dev | prod（默认 dev）
- LOG_LEVEL DEBUG | INFO | WARNING | ERROR（默认 INFO）
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_structlog(env: str | None = None,
                        level: str | None = None) -> None:
    """统一配置 structlog + 底层 logging（可重复调用，以最后一次为准）。"""
    env = (env or os.getenv("LOG_ENV", "dev")).lower()
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    # 1. 压制第三方库噪音（保留 WARNING 以上）
    for name in ("sqlalchemy", "httpx", "urllib3", "botocore", "minio"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 2. 处理器流水线（两模共享前段）
    shared_processors = [
        structlog.contextvars.merge_contextvars,   # 合并 contextvars（预留 request_id 贯穿）
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if env == "prod":
        # 生产：单行 JSON，机器可解析
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        # 开发：彩色控制台，人眼可读
        # （不使用 RichTracebackFormatter：那需要可选依赖 rich，锁文件未含）
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]

    # 3. structlog 配置：过滤级别用 LOG_LEVEL（NOTSET 会让级别设置失效）
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 4. 同步标准库 logging（uvicorn 等第三方日志走这里，格式透传 message）
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level_value,
    )


def get_logger(name: str):
    """便捷入口：structlog.get_logger 的项目内别名。"""
    return structlog.get_logger(name)
