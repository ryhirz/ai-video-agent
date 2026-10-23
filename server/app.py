"""FastAPI 路由层（全栈层架构 · 四的第一步）。

POST   /render                  提交渲染任务（202）
GET    /render/{id}             查询状态/进度
POST   /render/{id}/cancel      软取消（PENDING/RUNNING 有效）
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
from pydantic import BaseModel, Field

from ai_video_agent.registry import capability_manifest

from .executor import RenderExecutor, new_task_id
from .models import TaskDB, utcnow
from .state import TaskState
from .storage import LocalStorage

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
    storage = LocalStorage(storage_root)
    executor = RenderExecutor(db, storage, tmp_root)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        executor.shutdown()

    app = FastAPI(title="ai-video-agent render API", version="0.1.0",
                  lifespan=lifespan)
    app.state.db = db

    @app.post("/render", status_code=202)
    def create_render_task(req: RenderRequest) -> Dict[str, Any]:
        task_id = new_task_id()
        db.create(task_id, req.user_id, req.timeline, req.options,
                  req.webhook_url)
        executor.submit(task_id, req.timeline)
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
            executor.request_cancel(task_id)      # 标志位：worker 检查点生效
            db.update(task_id, state=TaskState.CANCELLED, finished_at=utcnow())
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

    return app


app = create_app()
