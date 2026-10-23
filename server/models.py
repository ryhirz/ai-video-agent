"""RenderTask 持久层（全栈层第二步：sqlite3 → SQLAlchemy 2.0）。

表结构 / 字段名与第一步完全一致（render_tasks 单表），新增两个二级索引：
  idx_state_created(state, created_at) —— 进行中任务扫描 / 僵尸任务检查
  idx_user(user_id)                    —— 按用户查历史任务
对外 API（create / get / update / utcnow / new_task_id）签名不变，
app.py / 任务体零改动迁移。每次操作独立短事务 + 进程内写锁 ——
SQLite 文件锁覆盖「1 worker + API 线程」的写竞争，与第一步策略一致。
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Float, Index, Text, create_engine, func, insert, select
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            sessionmaker)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class RenderTask(Base):
    """ORM 模型：字段与第一步 sqlite3 DDL 一致，仅二级索引为新增。"""

    __tablename__ = "render_tasks"
    __table_args__ = (
        Index("idx_state_created", "state", "created_at"),
        Index("idx_user", "user_id"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    timeline: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[str] = mapped_column(Text, nullable=False)
    result_url: Mapped[Optional[str]] = mapped_column(Text)
    error_msg: Mapped[Optional[str]] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[Optional[str]] = mapped_column(Text)
    finished_at: Mapped[Optional[str]] = mapped_column(Text)
    webhook_url: Mapped[Optional[str]] = mapped_column(Text)


class TaskDB:
    """render_tasks 的最小 DAL：CRUD + 白名单状态更新（线程安全）。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"timeout": 10.0},   # 等价 sqlite3 版 timeout
            future=True)
        self._session = sessionmaker(self._engine, expire_on_commit=False)
        Base.metadata.create_all(self._engine)

    def create(self, task_id: str, user_id: str, timeline: Dict[str, Any],
               options: Dict[str, Any], webhook_url: Optional[str]) -> None:
        with self._lock, self._session.begin() as session:
            session.execute(insert(RenderTask).values(
                id=task_id, user_id=user_id, state="pending",
                timeline=json.dumps(timeline, ensure_ascii=False),
                options=json.dumps(options, ensure_ascii=False),
                progress=0.0, created_at=utcnow(), webhook_url=webhook_url))

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(RenderTask).where(RenderTask.id == task_id)
        with self._session.begin() as session:
            row = session.execute(stmt).scalar_one_or_none()
        return self._to_dict(row) if row else None

    def update(self, task_id: str, **fields: Any) -> None:
        """白名单字段更新；timeline/options 需 JSON 序列化。"""
        allowed = {"state", "result_url", "error_msg", "progress",
                   "started_at", "finished_at", "timeline", "options"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不允许更新的字段: {unknown}")
        vals: Dict[str, Any] = {}
        for k, v in fields.items():
            if k in ("timeline", "options") and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            vals[k] = v
        with self._lock, self._session.begin() as session:
            session.query(RenderTask).filter_by(id=task_id).update(vals)

    def counts_by_state(self) -> Dict[str, int]:
        """按 state 分组计数（/metrics 队列深度 Gauge 的 DB 权威数据源）."""
        stmt = select(RenderTask.state, func.count()).group_by(RenderTask.state)
        with self._session.begin() as session:
            rows = session.execute(stmt).all()
        return {state: n for state, n in rows}

    @staticmethod
    def _to_dict(row: RenderTask) -> Dict[str, Any]:
        d = {c.name: getattr(row, c.name) for c in RenderTask.__table__.columns}
        d["timeline"] = json.loads(d["timeline"])
        d["options"] = json.loads(d["options"])
        return d


def new_task_id() -> str:
    return str(uuid.uuid4())
