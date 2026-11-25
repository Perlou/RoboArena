"""策略库。后端按名称注册；外部策略实现 Policy 接口并注册一个构造函数即可参加评测。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from roboarena.core.errors import NotFoundError
from roboarena.policies.api import Policy
from roboarena.policies.builtin import RandomPolicy
from roboarena.tasks.api import TaskSpec

__all__ = ["Policy", "build_policy", "list_policy_backends", "register_policy_backend"]

# 构造函数：(任务规格, 构造参数, 检查点路径或 None, 名称) → 策略
PolicyBuilder = Callable[[TaskSpec, Mapping[str, Any], Path | None, str], Policy]


def _build_random(spec: TaskSpec, params: Mapping[str, Any], checkpoint: Path | None, name: str) -> Policy:
    return RandomPolicy(spec, chunk=int(params.get("chunk", 1)), name=name)


_BACKENDS: dict[str, PolicyBuilder] = {"random": _build_random}


def register_policy_backend(backend: str, builder: PolicyBuilder) -> None:
    _BACKENDS[backend] = builder


def list_policy_backends() -> list[str]:
    return sorted(_BACKENDS)


def build_policy(
    backend: str, spec: TaskSpec, params: Mapping[str, Any], checkpoint: Path | None, name: str
) -> Policy:
    if backend not in _BACKENDS:
        raise NotFoundError(f"没有名为 {backend} 的策略后端。已注册：{'、'.join(list_policy_backends())}")
    return _BACKENDS[backend](spec, params, checkpoint, name)
