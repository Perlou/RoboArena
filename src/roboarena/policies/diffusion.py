"""Diffusion Policy（Chi et al. 2023，arXiv:2303.04137）的小型版本：用扩散模型生成一段动作。

行为克隆对同一个画面只能给出一个答案；演示里"从左边绕"和"从右边绕"都出现时，它会学成从中间撞过去。
扩散模型学的是整个动作分布：从噪声出发，一步步去噪成一段动作，每次采样落在其中一种做法上。

    训练    取干净的动作 a₀，随机时刻 t，加噪 a_t = √ᾱ_t·a₀ + √(1−ᾱ_t)·ε，让网络由 (a_t, t, 画面) 预测 ε
    采样    a_T ~ N(0, I)，用 DDIM 走 sample_steps 步回到 a₀

和原论文的差别：去噪网络是带残差的全连接网络（原论文用一维卷积 U-Net 或 Transformer），编码器与行为克隆相同。
噪声日程是余弦日程（Nichol & Dhariwal 2021，arXiv:2102.09672）。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

from roboarena.policies.modules import make_encoder, stack_frames

TIME_DIM = 64


def cosine_alphas_cumprod(n: int) -> Tensor:
    """余弦日程的 ᾱ_t，t = 0 … n−1。从接近 1 单调降到接近 0。"""
    steps = torch.arange(n + 1, dtype=torch.float64) / n
    f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    betas = (1.0 - f[1:] / f[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


def time_embedding(t: Tensor, dim: int = TIME_DIM) -> Tensor:
    """正弦位置编码：[B] → [B, dim]。"""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angle = t.float()[:, None] * freq[None]
    return torch.cat([angle.sin(), angle.cos()], dim=1)


class ResBlock(nn.Module):
    """h ← h + f(h, 条件)。条件（画面特征 + 时刻）在每一层都重新注入。"""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return h + self.net(torch.cat([self.norm(h), cond], dim=1))


class DiffusionNet(nn.Module):
    alphas_cumprod: Tensor

    def __init__(
        self,
        action_dim: int,
        obs_frames: int,
        chunk: int,
        width: int,
        hidden: int,
        diffusion_steps: int = 100,
        sample_steps: int = 10,
    ) -> None:
        super().__init__()
        self.obs_frames, self.chunk, self.action_dim = obs_frames, chunk, action_dim
        self.diffusion_steps, self.sample_steps = diffusion_steps, sample_steps
        self.encoder = make_encoder(obs_frames, width)
        self.cond_proj = nn.Sequential(nn.Linear(8 * width + TIME_DIM, hidden), nn.SiLU())
        self.in_proj = nn.Linear(chunk * action_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(3)])
        self.out = nn.Linear(hidden, chunk * action_dim)
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(diffusion_steps))

    def encode(self, frames: Tensor) -> Tensor:
        """frames [B, K, H, W, 3] uint8 → 画面特征 [B, 8·width]。采样时只算一次。"""
        return self.encoder(stack_frames(frames))

    def denoise(self, noisy: Tensor, t: Tensor, obs: Tensor) -> Tensor:
        """noisy [B, chunk, A]，t [B]，obs [B, 8·width] → 预测的噪声 [B, chunk, A]。"""
        cond = self.cond_proj(torch.cat([obs, time_embedding(t)], dim=1))
        h = self.in_proj(noisy.flatten(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).reshape(noisy.shape)

    def loss(self, frames: Tensor, target: Tensor, rng: np.random.Generator) -> Tensor:
        """时刻和噪声都从 rng 取：训练的全部随机性来自同一个发生器，续训才能逐位复现。"""
        b = len(target)
        t = torch.from_numpy(rng.integers(0, self.diffusion_steps, size=b)).to(target.device)
        noise = torch.from_numpy(rng.standard_normal(target.shape, dtype=np.float32)).to(target.device)
        ac = self.alphas_cumprod[t].reshape(b, 1, 1)
        noisy = ac.sqrt() * target + (1.0 - ac).sqrt() * noise
        return torch.nn.functional.mse_loss(self.denoise(noisy, t, self.encode(frames)), noise)

    @torch.no_grad()
    def sample(self, frames: Tensor, noise: Tensor) -> Tensor:
        """确定性的 DDIM 采样：给定初始噪声 [B, chunk, A]，走 sample_steps 步，返回归一化的动作。"""
        obs = self.encode(frames)
        x = noise
        ts = (
            torch.linspace(self.diffusion_steps - 1, 0, self.sample_steps, device=noise.device).round().long()
        )
        for i, t in enumerate(ts):
            ac = self.alphas_cumprod[t]
            eps = self.denoise(x, t.expand(len(x)), obs)
            x0 = ((x - (1.0 - ac).sqrt() * eps) / ac.sqrt()).clamp(-1.0, 1.0)  # 动作归一化后就在这个范围里
            if i == len(ts) - 1:
                return x0
            eps = (x - ac.sqrt() * x0) / (1.0 - ac).sqrt()  # 截断之后重新算一遍噪声，保持两者自洽
            ac_next = self.alphas_cumprod[ts[i + 1]]
            x = ac_next.sqrt() * x0 + (1.0 - ac_next).sqrt() * eps
        return x

    def predict(self, frames: Tensor, rng: np.random.Generator) -> Tensor:
        shape = (len(frames), self.chunk, self.action_dim)
        noise = torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)).to(frames.device)
        return self.sample(frames, noise)
