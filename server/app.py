"""FastAPI 路由层（全栈层第二步：Celery + Redis + DB 查询态取消）。

POST   /render                  提交渲染任务（202）→ render_video.delay
GET    /render/{id}             查询状态/进度
POST   /render/{id}/cancel      软取消（DB 查询态：只置 state=CANCELLED，
                                worker 四个检查点查 DB 生效，终态幂等）
GET    /render/{id}/download    下载成片（仅 SUCCESS）
GET    /manifest                能力清单（前端动态渲染参数表单）

认证（可选）：环境变量 API_KEY 设置后，/render* 四条业务路由要求请求头
X-API-Key 匹配（401 拒绝）；未设置 = 完全关闭认证，行为与历史版本一致。
/manifest 与 /app 静态页保持公开（表单渲染与页面加载不受影响）。

create_app(db_path, storage_root, tmp_root) 工厂便于测试注入临时目录；
模块级 `app = create_app()` 供 `uvicorn server.app:app` 直接启动。
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from structlog.contextvars import bind_contextvars, clear_contextvars

from ai_video_agent.registry import capability_manifest

from .logging import configure_structlog, get_logger
from .metrics import (APP_INFO, collect_metrics_middleware, metrics_response,
                      refresh_db_gauges)
from .models import TaskDB, new_task_id, utcnow
from .state import TaskState
from .storage import MinIOStorage, storage_from_env
from .tasks import celery_app, configure_runtime, get_runtime, render_video

# 日志初始化（模块级一次）：uvicorn server.app:app 的 API 进程入口；
# LOG_ENV=prod → JSON，dev（默认）→ 彩色文本
configure_structlog()

_DEFAULT_ROOT = Path(__file__).resolve().parent / "server_data"


class RenderRequest(BaseModel):
    timeline: Dict[str, Any]
    options: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "local"
    webhook_url: Optional[str] = None


def create_app(db_path: str | Path | None = None,
               storage_root: str | Path | None = None,
               tmp_root: str | Path | None = None) -> FastAPI:
    db_path = Path(db_path or _DEFAULT_ROOT / "tasks.db")
    storage_root = Path(storage_root or _DEFAULT_ROOT / "renders")
    tmp_root = Path(tmp_root or _DEFAULT_ROOT / "tmp")
    for p in (db_path.parent, storage_root, tmp_root):
        p.mkdir(parents=True, exist_ok=True)

    db = TaskDB(str(db_path))
    # 装配点（第四步）：MINIO_ENDPOINT 环境变量存在 → MinIO，否则本地 FS
    storage = storage_from_env(storage_root)
    # eager / API 进程内：任务体直接使用本 app 的资源实例（ADR-1）
    configure_runtime(db, storage, tmp_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(title="ai-video-agent render API", version="0.2.0",
                  lifespan=lifespan)
    app.state.db = db

    # ---- request_id 贯穿（结构化日志闭环，第十九节）----
    # 每请求生成/透传 X-Request-ID 并注入 structlog 上下文：
    # 客户端/网关自带 X-Request-ID 则透传（跨服务链路追踪），否则 UUID v4；
    # 任何后续 structlog 打点自动携带 request_id，无需手动传。
    _log = get_logger(__name__)

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        clear_contextvars()   # 协程复用场景防上下文泄漏
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        bind_contextvars(request_id=request_id)
        _start = time.perf_counter()
        response = await call_next(request)
        # 访问日志：每请求一行（含 request_id），业务代码零打点也可观测
        _log.info("http_request",
                  method=request.method,
                  path=request.url.path,
                  status=response.status_code,
                  duration_ms=round((time.perf_counter() - _start) * 1000, 1))
        response.headers["X-Request-ID"] = request_id
        return response

    # ---- Prometheus 指标采集（可观测性：请求计数/耗时/并发）----
    # 在 request_id 中间件之后注册（栈内层），度量真实下游耗时；
    # /metrics 端点自身排除，防止自采集递归打点。
    collect_metrics_middleware(app)
    APP_INFO.info({"version": app.version, "app": "ai-video-agent"})
    # ---- 可选认证：API_KEY 环境变量在 create_app 时读取一次 ----
    # 未设置 → _expected_key 为 None，依赖直接放行（零开销、向后兼容）
    _expected_key = os.environ.get("API_KEY") or None

    def require_api_key(
            x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
            api_key: Optional[str] = None) -> None:
        """/render* 业务路由的认证依赖：Key 未配置时恒放行。

        同时接受 X-API-Key 请求头与 api_key 查询参数 —— 后者服务于
        浏览器无法携带自定义头的场景（video.src / <a download>）。
        """
        if _expected_key is not None and _expected_key not in (x_api_key,
                                                               api_key):
            raise HTTPException(401, "Invalid or missing API key")

    _guard = [Depends(require_api_key)] if _expected_key is not None else []

    @app.post("/render", status_code=202, dependencies=_guard)
    def create_render_task(req: RenderRequest) -> Dict[str, Any]:
        task_id = new_task_id()
        db.create(task_id, req.user_id, req.timeline, req.options,
                  req.webhook_url)
        render_video.delay(task_id, req.timeline)   # eager 模式下同步执行
        return {"task_id": task_id, "state": TaskState.PENDING.value}

    @app.get("/render/{task_id}", dependencies=_guard)
    def get_task_status(task_id: str) -> Dict[str, Any]:
        task = db.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        return {"task_id": task["id"], "state": task["state"],
                "progress": task["progress"], "result_url": task["result_url"],
                "error_msg": task["error_msg"]}

    @app.post("/render/{task_id}/cancel", dependencies=_guard)
    def cancel_task(task_id: str) -> Dict[str, Any]:
        task = db.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task["state"] in (TaskState.PENDING.value, TaskState.RUNNING.value):
            # DB 查询态（ADR-2）：不接触 worker，标志就是 state 本身
            db.update(task_id, state=TaskState.CANCELLED, finished_at=utcnow())
        # 终态幂等：直接返回当前状态，不改库
        return {"task_id": task_id, "state": db.get(task_id)["state"]}

    @app.get("/render/{task_id}/download", dependencies=_guard)
    def download_result(task_id: str) -> FileResponse:
        task = db.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task["state"] != TaskState.SUCCESS.value or not task["result_url"]:
            raise HTTPException(400, "Task not completed")
        file = storage.path_of(task["result_url"])
        if not file.exists():
            raise HTTPException(410, "Result file expired")
        return FileResponse(file, media_type="video/mp4",
                            filename=task["result_url"])

    @app.get("/manifest")
    def get_manifest() -> Dict[str, Any]:
        return capability_manifest()

    @app.get("/health")
    def health() -> Dict[str, str]:
        """Liveness/readiness 探针端点（可观测性起步）。

        保持公开：容器编排（compose/K8s）探针无法携带 API Key；
        仅返回进程可达性，不探 DB/broker 深度依赖（就绪细节留待
        后续 /ready 按需扩展）。
        """
        return {"status": "ok", "version": app.version}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus 抓取端点（公开：抓取器无法携带 API Key，与 /health 同理）."""
        refresh_db_gauges(app.state.db)   # DB 权威队列深度（scrape 时快照）
        body, content_type = metrics_response()
        return Response(content=body, media_type=content_type)

    @app.get("/ready")
    def ready() -> Response:
        """Readiness 深度探针：DB / Redis broker / MinIO 连通性。

        与 /health（liveness，只探进程可达）互补：依赖不可用时应
        从负载均衡摘除流量而非重启容器。恒公开（探针无法携带
        API Key，与 /health 同理）。MinIO 是可选依赖——未配置时
        标记 skipped 而非失败。
        """
        deps: Dict[str, str] = {}
        ok = True

        # DB（权威状态源）：真实执行一次分组查询，可读即 SQLite 可用
        try:
            app.state.db.counts_by_state()
            deps["db"] = "ok"
        except Exception as e:                  # noqa: BLE001 —— 探针吞异常降级
            deps["db"] = f"error: {e}"
            ok = False

        # Redis broker：真实建立一次连接（超时 2s 快速失败）
        try:
            conn = celery_app.connection()
            try:
                conn.ensure_connection(max_retries=1, interval_start=0,
                                       interval_step=0, timeout=2)
                deps["redis"] = "ok"
            finally:
                conn.release()
        except Exception as e:                  # noqa: BLE001
            deps["redis"] = f"error: {e}"
            ok = False

        # MinIO（可选）：配置了端点才检查；未配置 = skipped（非失败）
        storage = get_runtime().storage
        if isinstance(storage, MinIOStorage):
            try:
                storage.client.list_buckets()
                deps["minio"] = "ok"
            except Exception as e:              # noqa: BLE001
                deps["minio"] = f"error: {e}"
                ok = False
        else:
            deps["minio"] = "skipped"

        payload = {"status": "ready" if ok else "not_ready",
                   "dependencies": deps}
        return JSONResponse(payload, status_code=200 if ok else 503)

    # 前端阶段1（最小预览页）：/app 同源服务 static/index.html ——
    # 避免 file:// 打开时的跨域拦截；仅挂载静态文件，业务路由零改动
    _static_dir = Path(__file__).resolve().parent / "static"
    if _static_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(_static_dir), html=True),
                  name="frontend")

    return app


app = create_app()
