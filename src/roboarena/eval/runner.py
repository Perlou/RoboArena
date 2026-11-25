"""评测循环：让一个策略在一个仿真器里从一组初始状态各跑一个回合。

这里不出现任何具体的策略、仿真器或任务：真实仿真器和神经仿真器走的是同一段代码。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from roboarena.policies.api import Policy
from roboarena.tasks.api import InitState, Simulator


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    max_progress: float  # 回合内达到过的最大进度
    steps: int  # 首次成功时已执行的步数；未成功则等于步数上限


def evaluate(
    policy: Policy,
    sim: Simulator,
    inits: Sequence[InitState],
    *,
    max_steps: int,
    n_exec: int,
    rng: np.random.Generator,
) -> list[EpisodeResult]:
    """B = len(inits) 个回合并行推进。

    每次调用策略得到一段动作 [B, chunk, A]，执行其中前 n_exec 个，再重新调用策略。
    已成功的回合继续跟着批一起步进，但结果不再更新：批的形状不变，实现简单，代价是少量多余计算。
    """
    if not 1 <= n_exec <= policy.chunk:
        raise ValueError(f"n_exec = {n_exec} 应在 1 到策略的分块长度 {policy.chunk} 之间")
    n = len(inits)
    frames = sim.reset(inits)  # [B, H, W, 3]
    history = np.repeat(frames[:, None], policy.obs_frames, axis=1)  # [B, obs_frames, H, W, 3]
    done = np.zeros(n, dtype=bool)
    best = np.zeros(n, dtype=np.float32)
    steps = np.full(n, max_steps, dtype=np.int64)

    t = 0
    while t < max_steps and not done.all():
        chunk = np.asarray(policy.act(history, rng))
        if chunk.ndim != 3 or chunk.shape[:2] != (n, policy.chunk):
            raise ValueError(f"策略 {policy.name} 输出的形状是 {chunk.shape}，应为 [{n}, {policy.chunk}, A]")
        for k in range(n_exec):
            if t >= max_steps or done.all():
                break
            result = sim.step(chunk[:, k])
            t += 1
            active = ~done
            best[active] = np.maximum(best[active], result.progress[active])
            newly = active & result.success
            steps[newly] = t
            done |= newly
            history = np.concatenate([history[:, 1:], result.frames[:, None]], axis=1)

    return [EpisodeResult(bool(done[i]), float(best[i]), int(steps[i])) for i in range(n)]


def evaluate_in_batches(
    policy: Policy,
    make_sim: Callable[[], Simulator],
    inits: Sequence[InitState],
    *,
    max_steps: int,
    n_exec: int,
    batch: int,
    make_rng: Callable[[int], np.random.Generator],
) -> list[EpisodeResult]:
    """把初始状态切成每批 batch 个依次评测。make_rng(起始下标) 给每一批一个独立的发生器。

    注意：带随机性的策略，结果与 batch 的取值有关（随机数的消耗顺序不同）。batch 记在运行的配置里。
    """
    sim = make_sim()
    try:
        results: list[EpisodeResult] = []
        for start in range(0, len(inits), batch):
            part = inits[start : start + batch]
            results += evaluate(policy, sim, part, max_steps=max_steps, n_exec=n_exec, rng=make_rng(start))
        return results
    finally:
        sim.close()
