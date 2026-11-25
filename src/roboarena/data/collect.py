"""在真实仿真器里采集带标签的回合。

只用人类演示训练的世界模型，在差策略到达的状态上会失真，所以要补两类数据：
    replay       把演示的动作序列加上噪声，从随机的初始状态开始执行（运动像人，但结果各不相同）
    random_walk  在动作范围内随机取路标点，沿折线移动（覆盖演示里没有的状态）
两者都是开环的：动作序列事先定好，再交给仿真器执行。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from roboarena.tasks.api import Episode, InitState, Simulator, Task, TaskSpec


def noisy_replay_actions(
    spec: TaskSpec, actions: np.ndarray, noise: float, rng: np.random.Generator
) -> np.ndarray:
    """给动作序列 [T, A] 加高斯噪声。noise 是标准差占动作范围的比例。"""
    scale = noise * (spec.action_high - spec.action_low)
    noisy = actions + rng.normal(0.0, 1.0, size=actions.shape) * scale
    return np.clip(noisy, spec.action_low, spec.action_high).astype(np.float32)


def random_walk_actions(spec: TaskSpec, steps: int, rng: np.random.Generator) -> np.ndarray:
    """随机路标点之间线性插值，每段 3–15 步。返回 [steps, A]。"""
    current = rng.uniform(spec.action_low, spec.action_high)
    out: list[np.ndarray] = []
    while len(out) < steps:
        target = rng.uniform(spec.action_low, spec.action_high)
        n = int(rng.integers(3, 16))
        out += [current + (target - current) * (k + 1) / n for k in range(n)]
        current = target
    return np.asarray(out[:steps], dtype=np.float32)


def rollout(
    sim: Simulator,
    inits: Sequence[InitState],
    action_seqs: Sequence[np.ndarray],
    metas: Sequence[dict[str, Any]],
) -> list[Episode]:
    """B 个回合一起执行各自的动作序列 [T_i, A]。长度不同的，短的用最后一个动作补齐，结果再截回原长。"""
    lengths = [len(a) for a in action_seqs]
    horizon = max(lengths)
    padded = np.stack([np.concatenate([a, np.repeat(a[-1:], horizon - len(a), axis=0)]) for a in action_seqs])
    results = [sim.reset(inits)]
    results += [sim.step(padded[:, t]) for t in range(horizon)]
    episodes = []
    for i, (length, meta) in enumerate(zip(lengths, metas, strict=True)):
        steps = results[: length + 1]
        episodes.append(
            Episode(
                frames=np.stack([r.frames[i] for r in steps]),
                actions=np.asarray(action_seqs[i], dtype=np.float32),
                progress=np.array([r.progress[i] for r in steps], dtype=np.float32),
                success=np.array([r.success[i] for r in steps], dtype=bool),
                labels={
                    k: np.stack([r.labels[k][i] for r in steps]).astype(np.float32) for k in results[0].labels
                },
                meta=meta,
            )
        )
    return episodes


def collect(
    task: Task,
    inits: Sequence[InitState],
    action_seqs: Sequence[np.ndarray],
    metas: Sequence[dict[str, Any]],
    batch: int = 16,
) -> list[Episode]:
    """分批采集。每个回合的 meta 里补上序列化后的初始状态，之后可以据此重放核对。"""
    sim = task.make_real_sim()
    try:
        episodes: list[Episode] = []
        for start in range(0, len(inits), batch):
            part = slice(start, start + batch)
            part_metas = [
                {**m, "init": task.encode_init(i)} for m, i in zip(metas[part], inits[part], strict=True)
            ]
            episodes += rollout(sim, inits[part], action_seqs[part], part_metas)
        return episodes
    finally:
        sim.close()
