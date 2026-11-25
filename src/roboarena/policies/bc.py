"""行为克隆（behavior cloning）：从画面直接回归一段动作。

    画面 [B, K, H, W, 3] ──▶ 编码器（卷积 + 空间 softmax）[B, 8·width]
                         ──▶ 两层全连接 ──▶ 动作分块 [B, chunk, A]，归一化到 [-1, 1]

损失用 L1：演示在同一个状态下可能有不同的做法，L1 学到的是中位数，比 L2 的均值少一些"两边都不像"的动作。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from roboarena.policies.modules import make_encoder, stack_frames


class BCNet(nn.Module):
    def __init__(self, action_dim: int, obs_frames: int, chunk: int, width: int, hidden: int) -> None:
        super().__init__()
        self.obs_frames, self.chunk, self.action_dim = obs_frames, chunk, action_dim
        self.encoder = make_encoder(obs_frames, width)
        self.head = nn.Sequential(
            nn.Linear(8 * width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk * action_dim),
        )

    def forward(self, frames: Tensor) -> Tensor:
        """frames [B, K, H, W, 3] uint8 → 归一化的动作 [B, chunk, A]。"""
        out = self.head(self.encoder(stack_frames(frames)))
        return out.reshape(len(frames), self.chunk, self.action_dim)

    def loss(self, frames: Tensor, target: Tensor, rng: np.random.Generator) -> Tensor:
        """target 是归一化的动作 [B, chunk, A]。行为克隆的损失没有随机性，rng 不用。"""
        return torch.nn.functional.l1_loss(self(frames), target)

    def predict(self, frames: Tensor, rng: np.random.Generator) -> Tensor:
        return self(frames)
