"""任务插件。按名称注册；实现模块在第一次用到时才导入，这样不用的任务不会带进它的依赖。"""

from __future__ import annotations

import importlib

from roboarena.core.errors import NotFoundError
from roboarena.tasks.api import Episode, InitState, Simulator, StepResult, Task, TaskSpec

__all__ = [
    "Episode",
    "InitState",
    "Simulator",
    "StepResult",
    "Task",
    "TaskSpec",
    "get_task",
    "list_tasks",
    "register_task",
]

# 名称 → "模块:工厂函数"
_FACTORIES: dict[str, str] = {
    "pusht": "roboarena.tasks.pusht:make_task",
    "toy": "roboarena.tasks.toy:make_task",
}


def register_task(name: str, factory: str) -> None:
    _FACTORIES[name] = factory


def list_tasks() -> list[str]:
    return sorted(_FACTORIES)


def get_task(name: str) -> Task:
    if name not in _FACTORIES:
        raise NotFoundError(f"没有名为 {name} 的任务。已注册：{'、'.join(list_tasks())}")
    module, attr = _FACTORIES[name].split(":")
    return getattr(importlib.import_module(module), attr)()
