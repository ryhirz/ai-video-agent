"""Prometheus 指标定义与采集中间件（可观测性：/metrics）.

指标清单（进程级单例，模块导入时向默认 REGISTRY 注册一次——
create_app 可被多次调用，指标不可重复注册）：
- http_requests_total          Counter    请求总数（method/path/status 分桶）
- http_request_duration_seconds Histogram 请求耗时（1ms~5s 自定义桶）
- http_requests_in_progress    Gauge      当前并发请求数（升/降语义，try/finally 兜底）
- app_info                     Info       应用元信息（version），供 Grafana 多实例过滤

基数控制：路径标签经 _metric_path 归一化——UUID 段（即 task_id）替换
为 {id}，剔除查询参数，防止动态路由把 label 基数打爆。
"""

from __future__ import annotations

import re
import time
from typing import Awaitable, Callable

from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               Info, generate_latest)

# ---- 指标定义（模块级一次） ----

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests",
    "HTTP 请求总数",
    labelnames=("method", "path", "status"),
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    labelnames=("method", "path"),
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "当前并发请求数",
    labelnames=("method", "path"),
)

APP_INFO = Info("app_info", "应用元信息（版本/名称）")

# ---- 路径基数归一化 ----

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _metric_path(request) -> str:
    """归一化路径标签：UUID 段 → {id}；不含查询参数（防标签基数爆炸）."""
    segments = request.url.path.split("/")
    normalized = [("{id}" if _UUID_RE.match(seg) else seg)
                  for seg in segments]
    return "/".join(normalized)


def collect_metrics_middleware(app) -> None:
    """给 FastAPI app 挂上指标采集中间件（在 request_id 中间件之后注册）.

    - /metrics 自身排除（防止自采集递归打点）
    - Gauge inc/dec 与 Counter inc 都走 try/except/finally：
      异常路径也正确回落（并发数不虚高、异常计入 500）
    """

    from starlette.requests import Request

    @app.middleware("http")
    async def _collect(request: Request,
                       call_next: Callable[[Request], Awaitable]):
        path = _metric_path(request)
        if path == "/metrics":
            return await call_next(request)

        method = request.method
        labels = {"method": method, "path": path}
        HTTP_REQUESTS_IN_PROGRESS.labels(**labels).inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path,
                                       status="500").inc()
            raise
        else:
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path,
                                       status=str(response.status_code)).inc()
            return response
        finally:
            HTTP_REQUESTS_IN_PROGRESS.labels(**labels).dec()
            HTTP_REQUEST_DURATION.labels(method=method,
                                         path=path).observe(
                                             time.perf_counter() - start)


def metrics_response() -> tuple[bytes, str]:
    """渲染 Prometheus 文本格式（供 /metrics 端点直接返回）."""
    return generate_latest(), CONTENT_TYPE_LATEST
