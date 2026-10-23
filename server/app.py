"""FastAPI 路由层（全栈层第二步：Celery + Redis + DB 查询态取消）。

POST   /render                  提交渲染任务（202）→ render_video.delay
GET    /render/{id}             查询状态/进度
POST   /render/{id}/cancel      软取消（DB 查询态：只置 state=CANCELLED，
                                worker 三个检查点查 DB 生效，终态幂等）
GET    /render/{id}/download    下载成片（仅 SUCCESS）
GET    /manifest                能力清单（前端动态渲染参数表单）

create_app(db_path, storage_root, tmp_root) 工厂便于测试注入临时目录；
模块级 `app = create_app()` 供 `uvicorn server.app:app` 直接启动。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_video_agent.registry import capability_manifest

from .models import TaskDB, new_task_id, utcnow
from .state import TaskState
from .storage import storage_from_env
from .tasks import configure_runtime, render_video

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

    @app.post("/render", status_code=202)
    def create_render_task(req: RenderRequest) -> Dict[str, Any]:
        task_id = new_task_id()
        db.create(task_id, req.user_id, req.timeline, req.options,
                  req.webhook_url)
        render_video.delay(task_id, req.timeline)   # eager 模式下同步执行
        return {"task_id": task_id, "state": TaskState.PENDING.value}

    @app.get("/render/{task_id}")
    def get_task_status(task_id: str) -> Dict[str, Any]:
        task = db.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        return {"task_id": task["id"], "state": task["state"],
                "progress": task["progress"], "result_url": task["result_url"],
                "error_msg": task["error_msg"]}

    @app.post("/render/{task_id}/cancel")
    def cancel_task(task_id: str) -> Dict[str, Any]:
        task = db.get(task_id)
        if task is None:
            raise HTTPException(404, "Task not found")
        if task["state"] in (TaskState.PENDING.value, TaskState.RUNNING.value):
            # DB 查询态（ADR-2）：不接触 worker，标志就是 state 本身
            db.update(task_id, state=TaskState.CANCELLED, finished_at=utcnow())
        # 终态幂等：直接返回当前状态，不改库
        return {"task_id": task_id, "state": db.get(task_id)["state"]}

    @app.get("/render/{task_id}/download")
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

    # 前端阶段1（最小预览页）：/app 同源服务 static/index.html ——
    # 避免 file:// 打开时的跨域拦截；仅挂载静态文件，业务路由零改动
    _static_dir = Path(__file__).resolve().parent / "static"
    if _static_dir.is_dir():
        app.mount("/app", StaticFiles(directory=str(_static_dir), html=True),
                  name="frontend")

    return app


app = create_app()
