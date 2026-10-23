"""Celery 任务层（全栈层第二步）。

架构决策（ADR-1 / ADR-2 / ADR-3，见 docs/全栈层/05-整体架构说明.md）：
- 测试策略：eager 为主（pytest 设 task_always_eager=True）+ 真队列冒烟 skipif
- 取消：DB 查询态 —— API 只把 state 置 CANCELLED，任务体在四个检查点查 DB
- 超时：soft_time_limit=540 捕获 SoftTimeLimitExceeded 优雅落 TIMEOUT；
        time_limit=600 硬杀兜底 + 任务体入口检查 created_at 距今超限直接
        落 TIMEOUT，防「硬杀 → reject 重回队列 → 再次渲染」死循环。

Worker 进程资源经环境变量注入（RENDER_DB_PATH / RENDER_STORAGE_ROOT /
RENDER_TMP_ROOT）；API/eager 进程内用 configure_runtime() 注入同一套实例。
"""
from __future__ import annotations

import os
import shutil
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

from ai_video_agent.compiler import compile as compile_timeline
from ai_video_agent.render import render
from ai_video_agent.timeline import Timeline

from .logging import configure_structlog
from .models import TaskDB, utcnow
from .state import TaskState
from .storage import LocalStorage, Storage, storage_from_env
from .webhooks import build_payload, fire_webhook

# 日志初始化（模块级一次）：celery -A server.tasks worker 是独立进程，
# 必须自行初始化（不能依赖 app 进程的配置）
configure_structlog()

_TIME_LIMIT_SECONDS = 600      # 与 conf.task_time_limit 保持一致
celery_app = Celery("ai_video_agent")
celery_app.conf.update(
    broker_url=os.environ.get("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
    task_default_queue="render",
    task_acks_late=True,               # 硬杀后消息重回队列
    task_reject_on_worker_lost=True,   # worker 丢任务时拒绝确认 → 重新投递
    task_time_limit=600,               # 硬杀兜底（架构书指定）
    task_soft_time_limit=540,          # 软超时 → 优雅落 TIMEOUT
    broker_connection_retry_on_startup=True,
)


class WorkerRuntime:
    """任务体依赖的资源集合（DB / 存储 / 临时目录），进程内单例。"""

    def __init__(self, db: TaskDB, storage: Storage, tmp_root: Path):
        self.db = db
        self.storage = storage
        self.tmp_root = Path(tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def cleanup_tmp(self, task_id: str) -> None:
        shutil.rmtree(self.tmp_root / task_id, ignore_errors=True)


_runtime: Optional[WorkerRuntime] = None


def configure_runtime(db: TaskDB, storage: Storage, tmp_root: str | Path) -> None:
    """API / eager 测试进程：create_app 把自己的资源实例注入任务体。"""
    global _runtime
    _runtime = WorkerRuntime(db, storage, tmp_root)


def get_runtime() -> WorkerRuntime:
    """Worker 子进程：按环境变量惰性初始化（默认与 app.py 的 _DEFAULT_ROOT 对齐）。"""
    global _runtime
    if _runtime is None:
        root = Path(__file__).resolve().parent / "server_data"
        db_path = os.environ.get("RENDER_DB_PATH", str(root / "tasks.db"))
        storage_root = os.environ.get("RENDER_STORAGE_ROOT", str(root / "renders"))
        tmp_root = os.environ.get("RENDER_TMP_ROOT", str(root / "tmp"))
        _runtime = WorkerRuntime(TaskDB(db_path), storage_from_env(storage_root),
                                 tmp_root)
    return _runtime


def _age_seconds(iso_created_at: str) -> float:
    created = datetime.fromisoformat(iso_created_at)
    now = datetime.now(timezone.utc)
    return (now - created).total_seconds()


def _is_cancelled_in_db(db: TaskDB, task_id: str) -> bool:
    """软取消唯一信号源：DB state == CANCELLED（ADR-2，不查任何内存标志）。"""
    row = db.get(task_id)
    return bool(row) and row["state"] == TaskState.CANCELLED.value


def _notify_webhook(db: TaskDB, task_id: str) -> None:
    """终态后按冻结契约回调 webhook_url（第四步）。

    无 webhook_url 跳过；回调失败不改变已落库的终态、不抛出。
    cancelled 路径的 state/finished_at 已由 API 写入 DB，直接读现值。
    """
    row = db.get(task_id)
    if row and row.get("webhook_url"):
        fire_webhook(row["webhook_url"], build_payload(row))


def _finish_terminal(db: TaskDB, rt: "WorkerRuntime", task_id: str, *,
                     state: TaskState, error_msg: str | None = None,
                     result_url: str | None = None,
                     progress: float | None = None) -> None:
    """终态收口：落库 + 清临时目录 + webhook 回调，所有终态路径共用。"""
    fields: Dict[str, Any] = {"state": state, "finished_at": utcnow()}
    if error_msg is not None:
        fields["error_msg"] = error_msg
    if result_url is not None:
        fields["result_url"] = result_url
    if progress is not None:
        fields["progress"] = progress
    db.update(task_id, **fields)
    rt.cleanup_tmp(task_id)
    _notify_webhook(db, task_id)


@celery_app.task(name="render_video", bind=True)
def render_video(self, task_id: str, timeline_json: Dict[str, Any]) -> None:
    """渲染任务体：executor._run 的状态推进原样迁移 + 超时衔接。

    状态推进：RUNNING → compile 预检 → render → storage.save → SUCCESS；
    异常全部落 FAILED + error_msg；四处软取消检查点查 DB（出队 / render 前 /
    render 后 / 产物入库前，出队检查必须先于 RUNNING 写入）；SoftTimeLimitExceeded
    优雅落 TIMEOUT。
    """
    rt = get_runtime()
    db = rt.db
    row = db.get(task_id)
    if row is None:
        return                                  # 防御：任务记录不存在

    # 硬杀兜底：重启/重回队列后若任务早已超过 time_limit，直接落 TIMEOUT
    if _age_seconds(row["created_at"]) > _TIME_LIMIT_SECONDS:
        _finish_terminal(db, rt, task_id, state=TaskState.TIMEOUT,
                         error_msg=f"Task exceeded {_TIME_LIMIT_SECONDS}s (entry check)")
        return

    # 检查点 0（出队）：必须先于 RUNNING 写入 —— 否则会把「排队中取消」
    # 写下的 CANCELLED 抹掉，取消信号丢失（DB 查询态的关键顺序约束）
    if _is_cancelled_in_db(db, task_id):
        rt.cleanup_tmp(task_id)
        _notify_webhook(db, task_id)     # API 已落 CANCELLED 终态，此处补回调
        return

    db.update(task_id, state=TaskState.RUNNING, started_at=utcnow())

    # 检查点 1：render 前
    if _is_cancelled_in_db(db, task_id):
        rt.cleanup_tmp(task_id)
        _notify_webhook(db, task_id)
        return

    try:
        tl = Timeline.from_dict(timeline_json)
        compile_timeline(tl)                    # 预检：非法 transform 等
    except SoftTimeLimitExceeded:
        _finish_terminal(db, rt, task_id, state=TaskState.TIMEOUT,
                         error_msg="Task exceeded soft time limit (540s)")
        return
    except Exception as e:                      # noqa: BLE001 —— 错误落库不抛
        _finish_terminal(db, rt, task_id, state=TaskState.FAILED,
                         error_msg=f"Compile error: {e}")
        return

    # 检查点 2：真正开始渲染前
    if _is_cancelled_in_db(db, task_id):
        rt.cleanup_tmp(task_id)
        _notify_webhook(db, task_id)
        return

    out_dir = rt.tmp_root / task_id
    last_progress_write = [0.0]     # monotonic 时刻；进度写库节流 ≥1s

    def _on_progress(p: float) -> None:
        now = _time.monotonic()
        if now - last_progress_write[0] < 1.0:
            return
        last_progress_write[0] = now
        db.update(task_id, progress=round(min(p, 1.0), 4))

    try:
        path, msg = render(tl, base_dir=str(out_dir),
                           on_progress=_on_progress,
                           cancel_check=lambda: _is_cancelled_in_db(db, task_id))
    except SoftTimeLimitExceeded:
        _finish_terminal(db, rt, task_id, state=TaskState.TIMEOUT,
                         error_msg="Task exceeded soft time limit (540s)")
        return
    except Exception as e:                      # noqa: BLE001
        _finish_terminal(db, rt, task_id, state=TaskState.FAILED,
                         error_msg=f"Render error: {e}")
        return
    if not path:
        # 中段取消：ffmpeg 已被 cancel_check 杀掉，DB 保持 CANCELLED
        # （API 早已写好终态；此处若落 FAILED 会把取消覆盖掉）
        if _is_cancelled_in_db(db, task_id):
            rt.cleanup_tmp(task_id)
            _notify_webhook(db, task_id)
            return
        _finish_terminal(db, rt, task_id, state=TaskState.FAILED,
                         error_msg=f"Render failed: {msg}")
        return

    # 检查点 3：产物入库前
    if _is_cancelled_in_db(db, task_id):
        rt.cleanup_tmp(task_id)
        _notify_webhook(db, task_id)
        return

    key = f"{task_id}.mp4"
    url = rt.storage.save(path, key)
    _finish_terminal(db, rt, task_id, state=TaskState.SUCCESS,
                     result_url=url, progress=1.0)
