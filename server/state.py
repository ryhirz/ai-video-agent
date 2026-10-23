"""任务状态机（全栈层架构 · 一）。

CANCELLED 是软取消：worker 在各阶段检查点看到标志位后停止；
正在运行的 ffmpeg 进程的两步terminate 留到第三步（进度解析 + proc.kill）。
"""
from __future__ import annotations

from enum import Enum


class TaskState(str, Enum):
    PENDING = "pending"      # 已入队，等待执行器领取
    RUNNING = "running"      # 正在编译/渲染
    SUCCESS = "success"      # 完成，结果可下载
    FAILED = "failed"        # 编译错误或 ffmpeg 异常
    CANCELLED = "cancelled"  # 用户主动取消（软取消）
    TIMEOUT = "timeout"      # 超时（单独状态，便于统计）


# 合法流转表（仅文档用途；执行器内部按此推进）
TRANSITIONS = {
    TaskState.PENDING: {TaskState.RUNNING, TaskState.CANCELLED},
    TaskState.RUNNING: {TaskState.SUCCESS, TaskState.FAILED,
                        TaskState.TIMEOUT, TaskState.CANCELLED},
}
