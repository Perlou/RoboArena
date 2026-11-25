"""需要训练的策略：按后端构造网络，把网络包装成 Policy。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from roboarena.core.device import get_device
from roboarena.core.errors import NotFoundError
from roboarena.policies.bc import BCNet
from roboarena.policies.diffusion import DiffusionNet
from roboarena.tasks.api import TaskSpec

# 后端 → 构造网络需要的参数名（除动作维度外）。登记策略时把它们存进注册表，加载时原样取出。
NET_PARAMS: dict[str, tuple[str, ...]] = {
    "bc": ("obs_frames", "chunk", "width", "hidden"),
    "diffusion": ("obs_frames", "chunk", "width", "hidden", "diffusion_steps", "sample_steps"),
}


class PolicyNet(Protocol):
    obs_frames: int
    chunk: int

    def loss(self, frames: Tensor, target: Tensor, rng: np.random.Generator) -> Tensor:
        """frames [B, K, H, W, 3] uint8，target 归一化的动作 [B, chunk, A] → 标量损失。"""
        ...

    def predict(self, frames: Tensor, rng: np.random.Generator) -> Tensor:
        """→ 归一化的动作 [B, chunk, A]。"""
        ...


def make_net(backend: str, action_dim: int, params: Mapping[str, Any]) -> nn.Module:
    if backend not in NET_PARAMS:
        raise NotFoundError(f"没有名为 {backend} 的可训练策略后端。可选：{'、'.join(NET_PARAMS)}")
    kwargs = {key: int(params[key]) for key in NET_PARAMS[backend]}
    return BCNet(action_dim, **kwargs) if backend == "bc" else DiffusionNet(action_dim, **kwargs)


class NetPolicy:
    def __init__(self, net: nn.Module, spec: TaskSpec, name: str, exec_noise: float = 0.0) -> None:
        self.name = name
        self.exec_noise = exec_noise  # 执行时给动作加的噪声，标准差占动作范围的比例；用来造出水平更低的策略
        self._low, self._high = spec.action_low, spec.action_high
        self._device = get_device()
        self._net: Any = net.to(self._device).eval()
        self.obs_frames: int = self._net.obs_frames
        self.chunk: int = self._net.chunk

    @torch.no_grad()
    def act(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        batch = torch.from_numpy(np.ascontiguousarray(frames)).to(self._device)
        out = self._net.predict(batch, rng).cpu().numpy()
        actions = (out + 1.0) / 2.0 * (self._high - self._low) + self._low
        if self.exec_noise > 0:
            actions = actions + rng.normal(0.0, 1.0, size=actions.shape) * self.exec_noise * (
                self._high - self._low
            )
        return np.clip(actions, self._low, self._high).astype(np.float32)


def build_net_policy(
    backend: str, spec: TaskSpec, params: Mapping[str, Any], checkpoint: Path | None, name: str
) -> NetPolicy:
    if checkpoint is None:
        raise ValueError(f"策略 {name} 没有检查点")
    net = make_net(backend, spec.action_dim, params)
    net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return NetPolicy(net, spec, name, float(params.get("exec_noise", 0.0)))
