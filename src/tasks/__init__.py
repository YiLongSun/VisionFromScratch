# 匯入子模組以註冊 Task
from . import detection, segmentation  # noqa: F401
from .base import TASKS, Task, build_task

__all__ = ["TASKS", "Task", "build_task"]
