"""采集策略回合：让已登记的策略在真实仿真器里闭环运行，把过程记录成回合。

这是数据的第三种来源。演示和开环采集的数据里没有"任务快完成和已经完成"的画面，
也没有"策略自己会走到的状态"；世界模型要评估这些策略，就得见过它们的行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from roboarena.config import Config
from roboarena.core.errors import RoboArenaError
from roboarena.core.log import get_logger
from roboarena.core.seeding import derive_seed, make_rng
from roboarena.data.rollouts import save_rollout_set
from roboarena.eval.runner import evaluate
from roboarena.policies import build_policy
from roboarena.policies.api import Policy
from roboarena.store.registry import checkpoint_path, get_artifact, list_artifacts
from roboarena.store.runs import start_run
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task
from roboarena.tasks.api import Episode, InitState, StepResult, Task

log = get_logger(__name__)


@dataclass(frozen=True)
class RolloutSummary:
    dir: Path
    per_policy: dict[str, tuple[int, int, int]]  # 策略 → (回合数, 成功的回合数, 帧数)
    run_id: str  # 这次采集的运行记录


def record_episodes(
    task: Task,
    policy: Policy,
    inits: list[InitState],
    *,
    max_steps: int,
    n_exec: int,
    extra_steps: int,
    rng: np.random.Generator,
) -> list[Episode]:
    """跑一批回合并记录。成功的回合保留到成功之后 extra_steps 步，其余保留到这一批结束。"""
    sim = task.make_real_sim()
    trace: list[tuple[np.ndarray | None, StepResult]] = []
    try:
        results = evaluate(
            policy, sim, inits, max_steps=max_steps, n_exec=n_exec, rng=rng, trace=trace, linger=extra_steps
        )
    finally:
        sim.close()
    steps_run = len(trace) - 1
    episodes = []
    for i, (init, result) in enumerate(zip(inits, results, strict=True)):
        length = min(result.steps + extra_steps, steps_run) if result.success else steps_run
        steps = [r for _, r in trace[: length + 1]]
        episodes.append(
            Episode(
                frames=np.stack([r.frames[i] for r in steps]),
                actions=np.stack([a[i] for a, _ in trace[1 : length + 1] if a is not None]).astype(
                    np.float32
                ),
                progress=np.array([r.progress[i] for r in steps], dtype=np.float32),
                success=np.array([r.success[i] for r in steps], dtype=bool),
                labels={
                    k: np.stack([r.labels[k][i] for r in steps]).astype(np.float32) for k in steps[0].labels
                },
                meta={"source": f"policy:{policy.name}", "labeled": True, "init": task.encode_init(init)},
            )
        )
    return episodes


def collect_rollouts(ws: Workspace, cfg: Config) -> RolloutSummary:
    """按 cfg.rollouts 采集并保存一个策略回合集，留下一条运行记录（用了哪些策略的哪份权重都记在里面）。

    初始状态的种子按（回合集名，策略名）派生，与评测集不相交。
    """
    rc = cfg.rollouts
    task = get_task(cfg.task.name)
    spec = task.spec
    names = rc.policies or [a.name for a in list_artifacts(ws, task=spec.name, kind="policy")]
    if not names:
        raise RoboArenaError(f"任务 {spec.name} 还没有登记任何策略")
    artifacts = [get_artifact(ws, name, kind="policy") for name in names]
    for artifact in artifacts:  # 前提在开始运行之前查完：不满足时直接报错，不留下一条失败的运行记录
        if artifact.task != spec.name:
            raise RoboArenaError(f"策略 {artifact.name} 属于任务 {artifact.task}，当前任务是 {spec.name}")
    max_steps = cfg.eval.max_steps or spec.max_steps

    with start_run(ws, "data", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=cfg.seed) as run:
        run.update_meta(rollouts=f"{spec.name}/{rc.name}", policies={a.name: a.sha256 for a in artifacts})
        episodes: list[Episode] = []
        per_policy: dict[str, tuple[int, int, int]] = {}
        for artifact in artifacts:
            name = artifact.name
            policy = build_policy(
                artifact.backend, spec, artifact.params, checkpoint_path(ws, artifact), name
            )
            inits = task.sample_inits(
                rc.per_policy, derive_seed(cfg.seed, f"rollouts:{rc.name}:{name}:inits")
            )
            mine: list[Episode] = []
            for start in range(0, len(inits), cfg.eval.batch):
                mine += record_episodes(
                    task,
                    policy,
                    inits[start : start + cfg.eval.batch],
                    max_steps=max_steps,
                    n_exec=min(cfg.eval.n_exec, policy.chunk),
                    extra_steps=rc.extra_steps,
                    rng=make_rng(cfg.seed, f"rollouts:{rc.name}:{name}:{start}"),
                )
            for i, ep in enumerate(mine):
                ep.meta.update(index=i, group=f"policy-{name}-{i}")
            per_policy[name] = (
                len(mine),
                sum(bool(e.success.any()) for e in mine),
                sum(len(e) + 1 for e in mine),
            )
            episodes += mine
            log.info("策略回合采集完成", extra={"policy": name, "episodes": len(mine)})
        out = save_rollout_set(ws.root, spec.name, rc.name, episodes)
        run.update_meta(per_policy=per_policy)
    return RolloutSummary(out, per_policy, run.id)
