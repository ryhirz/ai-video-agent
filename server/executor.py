"""渲染执行器（全栈层架构 · 三的第一步）。

ThreadPoolExecutor(max_workers=1) 占位 —— ffmpeg 吃 CPU，单并发；
第三步替换为 Celery worker 时，`_run` 的状态推进逻辑原样迁移。

软取消：API 把任务标 CANCELLED 并记入标志集合，`_run` 在三个检查点
（render 前 / render 后 / 产物入库前）发现后停止并清理临时产物。
第一步 render 是同步调用，无法中途打断进程 —— 流式取消留到第三步
（ffmpeg stderr 解析 + proc.terminate/kill）。
"""
from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from ai_video_agent.compiler import compile as compile_timeline
from ai_video_agent.render import render
from ai_video_agent.timeline import Timeline

from .models import TaskDB, utcnow
from .state import TaskState
from .storage import Storage


class RenderExecutor:
    def __init__(self, db: TaskDB, storage: Storage, tmp_root: str | Path,
                 workers: int = 1):
        self._db = db
        self._storage = storage
        self._tmp_root = Path(tmp_root)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self._cancelled: set[str] = set()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="render")

    # ---- 对外接口（与未来 Celery 版对齐） ----
    def submit(self, task_id: str, timeline: Dict) -> None:
        self._pool.submit(self._run, task_id, timeline)

    def request_cancel(self, task_id: str) -> None:
        """软取消：只记标志位；状态改写由 API 层负责。"""
        self._cancelled.add(task_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- 内部 ----
    def _is_cancelled(self, task_id: str) -> bool:
        return task_id in self._cancelled

    def _cleanup_tmp(self, task_id: str) -> None:
        shutil.rmtree(self._tmp_root / task_id, ignore_errors=True)

    def _run(self, task_id: str, timeline_json: Dict) -> None:
        db = self._db
        db.update(task_id, state=TaskState.RUNNING, started_at=utcnow())

        # 检查点 1：render 前
        if self._is_cancelled(task_id):
            self._cleanup_tmp(task_id)
            return

        try:
            tl = Timeline.from_dict(timeline_json)
            compile_timeline(tl)                 # 预检：非法 transform 等
        except Exception as e:                   # noqa: BLE001 —— 错误落库不抛
            db.update(task_id, state=TaskState.FAILED,
                      error_msg=f"Compile error: {e}", finished_at=utcnow())
            self._cleanup_tmp(task_id)
            return

        # 检查点 2：真正开始渲染前
        if self._is_cancelled(task_id):
            self._cleanup_tmp(task_id)
            return

        out_dir = self._tmp_root / task_id
        try:
            path, msg = render(tl, base_dir=str(out_dir))
        except Exception as e:                   # noqa: BLE001
            db.update(task_id, state=TaskState.FAILED,
                      error_msg=f"Render error: {e}", finished_at=utcnow())
            self._cleanup_tmp(task_id)
            return
        if not path:
            db.update(task_id, state=TaskState.FAILED,
                      error_msg=f"Render failed: {msg}", finished_at=utcnow())
            self._cleanup_tmp(task_id)
            return

        # 检查点 3：产物入库前
        if self._is_cancelled(task_id):
            self._cleanup_tmp(task_id)
            return

        key = f"{task_id}.mp4"
        url = self._storage.save(path, key)
        db.update(task_id, state=TaskState.SUCCESS, result_url=url,
                  progress=1.0, finished_at=utcnow())
        self._cleanup_tmp(task_id)
        self._cancelled.discard(task_id)


def new_task_id() -> str:
    return str(uuid.uuid4())
