"""RenderTask 持久层（全栈层架构 · 二）。

第一步用 sqlite3 标准库（零新依赖）；第二步引入 Celery 时升级为 SQLAlchemy，
表结构与字段名保持不变。每次操作开独立连接 —— 规避跨线程 check_same_thread，
SQLite 文件锁 + timeout 足以覆盖「1 个执行线程 + API 线程」的写竞争。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .state import TaskState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS render_tasks (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    state       TEXT NOT NULL,
    timeline    TEXT NOT NULL,
    options     TEXT NOT NULL,
    result_url  TEXT,
    error_msg   TEXT,
    progress    REAL NOT NULL DEFAULT 0.0,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    webhook_url TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskDB:
    """RenderTask 表的最小 DAL：CRUD + 状态更新，线程安全（连接内建锁语义）。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, task_id: str, user_id: str, timeline: Dict[str, Any],
               options: Dict[str, Any], webhook_url: Optional[str]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO render_tasks (id, user_id, state, timeline, options,"
                " progress, created_at, webhook_url) VALUES (?,?,?,?,?,?,?,?)",
                (task_id, user_id, TaskState.PENDING.value,
                 json.dumps(timeline, ensure_ascii=False),
                 json.dumps(options, ensure_ascii=False),
                 0.0, utcnow(), webhook_url))

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM render_tasks WHERE id = ?",
                               (task_id,)).fetchone()
        return self._to_dict(row) if row else None

    def update(self, task_id: str, **fields: Any) -> None:
        """白名单字段更新；timeline/options 需 JSON 序列化。"""
        allowed = {"state", "result_url", "error_msg", "progress",
                   "started_at", "finished_at", "timeline", "options"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不允许更新的字段: {unknown}")
        cols, vals = [], []
        for k, v in fields.items():
            if k in ("timeline", "options") and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            if isinstance(v, TaskState):
                v = v.value
            cols.append(f"{k} = ?")
            vals.append(v)
        vals.append(task_id)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE render_tasks SET {', '.join(cols)} WHERE id = ?",
                         vals)

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["timeline"] = json.loads(d["timeline"])
        d["options"] = json.loads(d["options"])
        return d
