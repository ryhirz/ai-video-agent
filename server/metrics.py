"""Prometheus 指标定义与采集中间件（可观测性：/metrics）.

指标清单（进程级单例，模块导入时向默认 REGISTRY 注册一次——
create_app 可被多次调用，指标不可重复注册）：
- http_requests_total          Counter    请求总数（method/path/status 分桶）
- http_request_duration_seconds Histogram 请求耗时（1ms~5s 自定义桶）
- http_requests_in_progress    Gauge      当前并发请求数（升/降语义，try/finally 兜底）
- app_info                     Info       应用元信息（version），供 Grafana 多实例过滤

worker 侧渲染指标（任务体薄壳埋点）：
- render_tasks_total           Counter    终态计数（outcome 分桶）
- render_duration_seconds      Histogram  渲染耗时（成功任务）
- render_active                Gauge      worker 正在执行的渲染任务数
- render_queue_depth / render_running_depth  Gauge  DB 权威队列深度
  （API 进程 /metrics 被 scrape 时快照刷新，见 refresh_db_gauges）

进程边界：prometheus 拉取模型是进程内的——eager/测试模式下任务与
API 同进程，/metrics 直接可见；生产双进程时 worker 须经
start_worker_metrics() 自带端点（compose 暴露 9101）。

基数控制：路径标签经 _metric_path 归一化——UUID 段（即 task_id）替换
为 {id}，剔除查询参数，防止动态路由把 label 基数打爆。
"""

from __future__ import annotations

import os
import re
import time
from typing import Awaitable, Callable

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry,
                               Counter, Gauge, Histogram, Info,
                               generate_latest)
from prometheus_client.exposition import start_http_server

# ---- multiprocess 模式初始化（必须在指标定义之前）----
# PROMETHEUS_MULTIPROC_DIR 在进程启动前注入（compose 共享卷）时，
# 指标值落共享目录的 mmap 文件——API 进程 /metrics 聚合全部进程；
# worker 无需自有 HTTP 端点。未设置 = 单进程模式（行为与旧版一致）。
_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR") or None
if _MULTIPROC_DIR:
    os.makedirs(_MULTIPROC_DIR, exist_ok=True)
MULTIPROC_ENABLED = bool(_MULTIPROC_DIR)

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

# worker 侧活跃数：multiprocess 模式下按进程求和（livesum）——
# 每个子进程各持一份 mmap 值，聚合端可见总并发；单进程模式参数被忽略
RENDER_ACTIVE = Gauge(
    "render_active",
    "worker 正在执行的渲染任务数",
    multiprocess_mode="livesum",
)

# Info 指标不被 multiprocess 模式支持（prometheus_client 已知限制）
# → 放独立私有 registry，metrics_response 手动拼接（两种模式通用）
_APP_INFO_REGISTRY = CollectorRegistry()
APP_VERSION = "0.2.0"
APP_INFO = Info("app_info", "应用元信息（版本/名称）",
                registry=_APP_INFO_REGISTRY)
# 模块级注入默认元信息（create_app 之外的进程——如 multiprocess 聚合
# 子进程——也能暴露版本；与 app.py 的 FastAPI(version=...) 保持一致）
APP_INFO.info({"version": APP_VERSION, "app": "ai-video-agent"})

# ---- worker 侧渲染指标（任务体薄壳埋点；eager 模式与 API 同进程，
#      生产双进程时 multiprocess 聚合或 worker 自有端点暴露） ----

RENDER_DURATION = Histogram(
    "render_duration_seconds",
    "渲染耗时（秒，成功任务；540 上界对应 soft_time_limit）",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 540),
)

RENDER_TOTAL = Counter(
    "render_tasks",
    "渲染任务终态数（outcome=success/failed/cancelled/timeout）",
    labelnames=("outcome",),
)

# ---- DB 权威队列深度（API 进程 /metrics 被 scrape 时快照刷新） ----

RENDER_QUEUE_DEPTH = Gauge(
    "render_queue_depth",
    "DB 层 pending（队列等待）任务数",
)

RENDER_RUNNING_DEPTH = Gauge(
    "render_running_depth",
    "DB 层 running 任务数",
)


def refresh_db_gauges(db) -> None:
    """从 DB 按状态计数刷新队列深度 Gauge（拉取时快照，无需 worker 广播）."""
    counts = db.counts_by_state()
    RENDER_QUEUE_DEPTH.set(counts.get("pending", 0))
    RENDER_RUNNING_DEPTH.set(counts.get("running", 0))


_worker_server_started = False


def start_worker_metrics(port: int | None = None) -> bool:
    """worker 进程暴露自有指标的 HTTP 端口（单进程模式用）。

    multiprocess 模式（PROMETHEUS_MULTIPROC_DIR 已设）下指标落共享
    目录、API /metrics 聚合时可见 → **跳过 HTTP 服务器**，返回 False。

    单进程模式下返回是否成功绑定；prefork 多并发时仅首个子进程绑定
    成功，其余 OSError 静默跳过（concurrency>1 请改用 multiprocess
    模式而非依赖此端口——单端口只能反映单个子进程）。
    """
    global _worker_server_started
    if MULTIPROC_ENABLED:
        _worker_server_started = True
        return False
    port = int(os.environ.get("WORKER_METRICS_PORT", "9101")) \
        if port is None else port
    try:
        start_http_server(port)
        _worker_server_started = True
        return True
    except OSError:
        return False

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
    """渲染 Prometheus 文本格式（供 /metrics 端点直接返回）.

    单进程模式：generate_latest(默认 REGISTRY)。
    multiprocess 模式：MultiProcessCollector 从共享目录聚合全部进程
    的 mmap 值（含 worker 并发子进程）；Info 指标（app_info）不被
    multiprocess 支持 → 从私有 registry 独立拼接。

    ⚠️ 已知限制（prometheus_client 固有）：进程死亡后其 mmap 文件
    残留会虚高计数——compose 滚动重部署时清理 prometheus_data 卷，
    或接受重启窗口内的轻微误差。
    """
    if MULTIPROC_ENABLED:
        from prometheus_client import multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry, path=_MULTIPROC_DIR)
        body = generate_latest(registry)
    else:
        body = generate_latest()
    body += generate_latest(_APP_INFO_REGISTRY)
    return body, CONTENT_TYPE_LATEST
