"""数据校验。每一项检查是一个函数：没问题返回 None，有问题返回一句说明。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from roboarena.tasks.api import Episode, Task, TaskSpec

Check = Callable[[TaskSpec, Episode], str | None]


def check_schema(spec: TaskSpec, ep: Episode) -> str | None:
    """T 个动作配 T + 1 帧，各数组的形状、类型与任务声明的一致。"""
    steps = len(ep.actions)
    expected: list[tuple[str, np.ndarray, tuple[int, ...], type]] = [
        ("frames", ep.frames, (steps + 1, *spec.frame_shape), np.uint8),
        ("actions", ep.actions, (steps, spec.action_dim), np.float32),
        ("progress", ep.progress, (steps + 1,), np.float32),
        ("success", ep.success, (steps + 1,), np.bool_),
    ]
    if steps < 1:
        return "回合里没有动作"
    if set(ep.labels) != set(spec.label_fields):
        return f"标签字段是 {sorted(ep.labels)}，任务声明的是 {sorted(spec.label_fields)}"
    expected += [
        (f"labels.{k}", ep.labels[k], (steps + 1, d), np.float32) for k, d in spec.label_fields.items()
    ]
    for name, array, shape, dtype in expected:
        if array.shape != shape or array.dtype != dtype:
            return f"{name} 是 {array.dtype}{list(array.shape)}，应为 {np.dtype(dtype)}{list(shape)}"
    return None


def check_action_range(spec: TaskSpec, ep: Episode) -> str | None:
    """动作是有限值，且在任务的动作范围内。"""
    if not np.isfinite(ep.actions).all():
        return "动作里有 NaN 或无穷"
    eps = 1e-4 * (spec.action_high - spec.action_low)
    if ((ep.actions < spec.action_low - eps) | (ep.actions > spec.action_high + eps)).any():
        return f"动作超出范围：最小 {ep.actions.min(axis=0)}，最大 {ep.actions.max(axis=0)}"
    return None


def check_values(spec: TaskSpec, ep: Episode) -> str | None:
    """进度在 0–1 之间（未知的填 NaN）；标签没有无穷；画面不是一片纯色。"""
    known = ep.progress[~np.isnan(ep.progress)]
    if ((known < 0) | (known > 1)).any():
        return f"进度超出 0–1：最小 {known.min()}，最大 {known.max()}"
    if any(np.isinf(v).any() for v in ep.labels.values()):
        return "标签里有无穷"
    if ep.frames.min() == ep.frames.max():
        return "画面是一片纯色"
    return None


def check_labels_complete(spec: TaskSpec, ep: Episode) -> str | None:
    """声称带标签的回合，进度和全部标签都不能有 NaN。"""
    if not ep.meta.get("labeled", False):
        return None
    if np.isnan(ep.progress).any() or any(np.isnan(v).any() for v in ep.labels.values()):
        return "回合标为带标签，但进度或标签里有 NaN"
    return None


CHECKS: dict[str, Check] = {
    "形状与类型": check_schema,
    "动作范围": check_action_range,
    "数值范围": check_values,
    "标签完整": check_labels_complete,
}


def check_episode(spec: TaskSpec, ep: Episode) -> dict[str, str | None]:
    """依次运行全部检查。形状不对时后面的检查没有意义，直接跳过。"""
    results: dict[str, str | None] = {}
    for name, check in CHECKS.items():
        results[name] = check(spec, ep)
        if name == "形状与类型" and results[name] is not None:
            break
    return results


def check_replay(task: Task, ep: Episode) -> str | None:
    """从记录的初始状态出发重新执行动作，画面应逐像素相同。

    这一项同时检查了三件事：动作与画面没有错位、文件没有损坏、仿真器是确定的。
    没有记录初始状态的回合（如人类演示）无法重放，返回 None。
    """
    if ep.meta.get("init") is None:
        return None
    sim = task.make_real_sim()
    try:
        first = sim.reset([task.decode_init(ep.meta["init"])])
        if not np.array_equal(first.frames[0], ep.frames[0]):
            return "重放的首帧与记录不同"
        for t, action in enumerate(ep.actions):
            result = sim.step(action[None])
            if not np.array_equal(result.frames[0], ep.frames[t + 1]):
                return f"重放到第 {t + 1} 步时画面与记录不同"
            if not np.isclose(result.progress[0], ep.progress[t + 1], atol=1e-6):
                return f"重放到第 {t + 1} 步时进度与记录不同"
    finally:
        sim.close()
    return None
