"""策略库：一组水平拉开的策略，供"神经仿真器里的排序是否可信"这个问题使用。

水平靠四种办法拉开，都不需要另外的算法：
    训练程度    基准策略训练过程中的若干快照        <base>-step<N>
    数据量      只用一部分演示重新训练              <base>-demos<N>
    执行噪声    给基准策略的动作加噪声              <base>-noise<σ>
    另一种算法  Diffusion Policy 及其训练快照       dp、dp-step<N>（zoo.diffusion = true 时）
    随机策略    下界                                random
"""

from __future__ import annotations

import copy

from roboarena.config import Config
from roboarena.policies.train import get_artifact_or_none, train_policy
from roboarena.store.registry import checkpoint_path, get_artifact, register_artifact
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task


def zoo_names(cfg: Config) -> list[str]:
    """按配置，策略库里应该有哪些策略。"""
    base = cfg.zoo.base
    names = [base, *(f"{base}-step{s}" for s in cfg.policy.snapshots if s < cfg.policy.steps)]
    names += [f"{base}-demos{n}" for n in cfg.zoo.n_demos]
    names += [f"{base}-noise{sigma:g}" for sigma in cfg.zoo.exec_noise]
    if cfg.zoo.diffusion:
        names += ["dp", *(f"dp-step{s}" for s in cfg.diffusion.snapshots if s < cfg.diffusion.steps)]
    return [*names, "random"]


def build_zoo(ws: Workspace, cfg: Config) -> tuple[list[str], list[str]]:
    """训练并登记策略库。已经登记的跳过，所以中途失败后重新运行会接着做。返回（全部名称，这次新做的名称）。"""
    base = cfg.zoo.base
    task = get_task(cfg.task.name).spec.name
    created: list[str] = []

    def missing(name: str) -> bool:
        return get_artifact_or_none(ws, name) is None

    if missing(base):
        train_policy(ws, cfg, base)
        created += [n for n in zoo_names(cfg) if n == base or n.startswith(f"{base}-step")]
    for n in cfg.zoo.n_demos:
        if missing(name := f"{base}-demos{n}"):
            variant = copy.deepcopy(cfg)
            variant.policy.n_demos, variant.policy.snapshots = n, []
            train_policy(ws, variant, name)
            created.append(name)
    base_artifact = get_artifact(ws, base, kind="policy")
    for sigma in cfg.zoo.exec_noise:
        if missing(name := f"{base}-noise{sigma:g}"):
            register_artifact(
                ws, task=task, name=name, kind="policy", backend=base_artifact.backend, run_id=base_artifact.run_id,
                checkpoint=checkpoint_path(ws, base_artifact), params={**base_artifact.params, "exec_noise": sigma},
            )  # fmt: skip
            created.append(name)
    if cfg.zoo.diffusion and missing("dp"):
        train_policy(ws, cfg, "dp", backend="diffusion")
        created += [n for n in zoo_names(cfg) if n == "dp" or n.startswith("dp-step")]
    if missing("random"):
        register_artifact(ws, task=task, name="random", kind="policy", backend="random", params={})
        created.append("random")
    return zoo_names(cfg), created
