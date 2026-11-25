"""策略网络共用的部件：画面编码器、动作归一化。"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpatialSoftmax(nn.Module):
    """[B, C, h, w] → [B, 2C]：对每个通道的激活做 softmax，再求坐标的期望（坐标范围 -1 到 1）。

    它把"这个特征在画面的哪里"直接变成坐标，适合输出本身就是位置的任务。
    """

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        prob = torch.softmax(x.reshape(b, c, h * w), dim=-1).reshape(b, c, h, w)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        ex = (prob.sum(dim=2) * xs).sum(dim=-1)  # [B, C]
        ey = (prob.sum(dim=3) * ys).sum(dim=-1)
        return torch.cat([ex, ey], dim=1)


def make_encoder(obs_frames: int, width: int) -> nn.Sequential:
    """4 层卷积（分辨率降到 1/4）+ 空间 softmax。输入 [B, 3·obs_frames, H, W]，输出 [B, 8·width]。"""

    def block(c_in: int, c_out: int, kernel: int, stride: int) -> list[nn.Module]:
        return [nn.Conv2d(c_in, c_out, kernel, stride, kernel // 2), nn.GroupNorm(8, c_out), nn.ReLU()]

    return nn.Sequential(
        *block(3 * obs_frames, width, 5, 2),
        *block(width, 2 * width, 3, 2),
        *block(2 * width, 4 * width, 3, 1),
        nn.Conv2d(4 * width, 4 * width, 3, 1, 1),
        SpatialSoftmax(),
    )


def stack_frames(frames: Tensor) -> Tensor:
    """[B, K, H, W, 3] uint8 → [B, 3K, H, W] float32，取值 -0.5 到 0.5：把 K 帧在通道上叠起来。"""
    b, k, h, w, _ = frames.shape
    return frames.permute(0, 1, 4, 2, 3).reshape(b, 3 * k, h, w).float() / 255.0 - 0.5


def normalize_actions(actions: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """环境坐标 → [-1, 1]。"""
    return (actions - low) / (high - low) * 2.0 - 1.0
