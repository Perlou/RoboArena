"""训练策略（行为克隆或 Diffusion Policy）：只用数据集训练部分里的演示回合。支持断点续训。

续训用的检查点（checkpoints/last.pt）保存模型、优化器、步数、采样用的随机数状态，以及 EMA 权重（如果开了）。
训练中的全部随机性来自同一个 NumPy 发生器：取哪些样本、画面平移多少、扩散的时刻与噪声。
所以在 CPU 上，中断后续训与不中断训练得到的权重逐位相同。
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboarena.config import Config, PolicyConfig, load_config
from roboarena.core.device import get_device
from roboarena.core.errors import ConflictError, NotFoundError, RoboArenaError
from roboarena.core.log import get_logger
from roboarena.core.seeding import derive_seed, make_rng
from roboarena.data.dataset import Dataset, load_dataset
from roboarena.data.episodes import load_episode
from roboarena.data.frames import FrameStore, random_shift
from roboarena.policies.modules import normalize_actions
from roboarena.policies.nets import NET_PARAMS, make_net
from roboarena.store.registry import Artifact, get_artifact, register_artifact
from roboarena.store.runs import Run, resume_run, start_run, truncate_metrics
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task

log = get_logger(__name__)

StepHook = Callable[[int], None]


def policy_config(cfg: Config, backend: str) -> PolicyConfig:
    """每个后端有自己的配置段：行为克隆用 [policy]，Diffusion Policy 用 [diffusion]。"""
    if backend not in NET_PARAMS:
        raise NotFoundError(f"没有名为 {backend} 的可训练策略后端。可选：{'、'.join(NET_PARAMS)}")
    return cfg.policy if backend == "bc" else cfg.diffusion


def demo_store(dataset: Dataset, split: str, limit: int = -1) -> FrameStore:
    """数据集某个子集里的演示回合。limit > 0 时只取前 limit 条（按清单顺序，所以少的是多的前缀）。"""
    entries = [e for e in dataset.entries(split) if e["source"] == "demo"]
    if limit > 0:
        entries = entries[:limit]
    if not entries:
        raise RoboArenaError(f"数据集的 {split} 部分没有演示回合")
    return FrameStore.from_episodes(load_episode(dataset.dir / e["file"]) for e in entries)


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """线性升温 warmup 步，之后余弦衰减到 0。直接由步数算出，续训时不需要恢复调度器。"""
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def ema_decay(step: int, target: float) -> float:
    """EMA 的衰减系数随步数升到 target：训练初期权重变化快，不让随机初始化的权重在平均里占太久。"""
    return min(target, (1.0 + step) / (10.0 + step))


def _save_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _cpu_state(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def get_artifact_or_none(ws: Workspace, name: str) -> Artifact | None:
    try:
        return get_artifact(ws, name)
    except NotFoundError:
        return None


def train_policy(
    ws: Workspace,
    cfg: Config,
    name: str,
    *,
    backend: str = "bc",
    resume: str | None = None,
    on_step: StepHook | None = None,
) -> Artifact:
    """训练一个策略并登记为 name；配置里 snapshots 列出的步数另外登记成 name-step<N>。

    resume 给出运行编号时，沿用那次运行保存的配置、名称和后端，从它最后一个检查点接着训练。
    on_step(步数) 在每步结束后调用，测试用它模拟中断。
    """
    if resume is not None:
        run_dir = ws.root / "runs" / resume
        if not (run_dir / "checkpoints" / "last.pt").exists():
            raise NotFoundError(f"运行 {resume} 没有可以续训的检查点")
        cfg = load_config(run_dir / "config.toml")
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        name, backend = meta["policy"], meta.get("backend", "bc")  # 加入后端之前的运行只有行为克隆
    pc = policy_config(cfg, backend)
    if get_artifact_or_none(ws, name) is not None:
        raise ConflictError(f"名称已被占用：{name}")

    task = get_task(cfg.task.name)
    spec = task.spec
    dataset = load_dataset(ws, spec.name, cfg.data.name)
    train, val = demo_store(dataset, "train", pc.n_demos), demo_store(dataset, "val")
    device = get_device()
    net_params = {key: getattr(pc, key) for key in NET_PARAMS[backend]}

    torch.manual_seed(derive_seed(cfg.seed, f"policy:{name}:init") % (2**31))
    net: Any = make_net(backend, spec.action_dim, net_params).to(device)
    ema: Any = copy.deepcopy(net).requires_grad_(False) if pc.ema > 0 else None
    optimizer = torch.optim.AdamW(net.parameters(), lr=pc.lr, weight_decay=pc.weight_decay)
    rng = make_rng(cfg.seed, f"policy:{name}:batches")
    low, high = torch.from_numpy(spec.action_low).to(device), torch.from_numpy(spec.action_high).to(device)

    def loss_of(frames: np.ndarray, actions: np.ndarray, loss_rng: np.random.Generator) -> torch.Tensor:
        target = normalize_actions(torch.from_numpy(actions).to(device), low, high)
        return net.loss(torch.from_numpy(frames).to(device), target, loss_rng)

    @torch.no_grad()
    def val_loss() -> float:
        """用训练中的权重（不是 EMA）在固定的验证批上算损失。"""
        net.eval()
        val_rng = make_rng(
            cfg.seed, f"policy:{name}:val"
        )  # 每次都从头开始：验证用的批固定，也不消耗训练的随机数
        losses = [
            loss_of(*val.policy_batch(val_rng, pc.batch, pc.obs_frames, pc.chunk), val_rng).item()
            for _ in range(pc.val_batches)
        ]
        net.train()
        return float(np.mean(losses))

    def register(run: Run, step: int, artifact_name: str) -> Artifact:
        """登记当前的权重；开了 EMA 时登记的是 EMA 权重。"""
        path = run.dir / "checkpoints" / f"step-{step}.pt"
        _save_atomic(_cpu_state(ema if ema is not None else net), path)
        return register_artifact(
            ws, task=spec.name, name=artifact_name, kind="policy", backend=backend, checkpoint=path, run_id=run.id,
            params={**net_params, "exec_noise": 0.0, "n_demos": train.n_episodes, "steps": step},
        )  # fmt: skip

    def loop(run: Run, step: int) -> Artifact:
        ckpt_dir = run.dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        started, started_step = time.perf_counter(), step
        net.train()
        while step < pc.steps:
            frames, actions = train.policy_batch(rng, pc.batch, pc.obs_frames, pc.chunk)
            frames = random_shift(frames, pc.aug_shift, rng)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, pc.lr, pc.warmup, pc.steps)
            loss = loss_of(frames, actions, rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    weight = 1.0 - ema_decay(step, pc.ema)
                    for averaged, current in zip(ema.parameters(), net.parameters(), strict=True):
                        averaged.lerp_(current, weight)
            step += 1

            if step % pc.log_every == 0 or step == pc.steps:
                run.log_metric(step, "loss", loss.item())
            if step in pc.snapshots and step != pc.steps:
                if get_artifact_or_none(ws, f"{name}-step{step}") is None:
                    register(run, step, f"{name}-step{step}")
            if step % pc.eval_every == 0 or step == pc.steps:
                run.log_metric(step, "val_loss", val_loss())
                run.log_metric(step, "steps_per_sec", (step - started_step) / (time.perf_counter() - started))
                state = {
                    "model": _cpu_state(net),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "rng": json.dumps(rng.bit_generator.state),
                }
                if ema is not None:
                    state["ema"] = _cpu_state(ema)
                _save_atomic(state, ckpt_dir / "last.pt")
            if on_step is not None:
                on_step(step)
        artifact = register(run, step, name)
        log.info(
            "策略训练完成",
            extra={"policy": name, "backend": backend, "steps": step, "demos": train.n_episodes},
        )
        return artifact

    if resume is None:
        with start_run(
            ws, "train_policy", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=cfg.seed,
            dataset_fp=dataset.fingerprint, device=str(device),
        ) as run:  # fmt: skip
            run.update_meta(policy=name, backend=backend, n_demos=train.n_episodes, n_samples=train.n_samples)
            return loop(run, 0)

    with resume_run(ws, resume) as run:
        state = torch.load(run.dir / "checkpoints" / "last.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = json.loads(state["rng"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        truncate_metrics(run.dir, state["step"])
        return loop(run, int(state["step"]))
