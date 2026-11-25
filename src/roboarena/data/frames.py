"""训练用的帧库：把一批回合的画面和动作首尾相接放进内存，按下标取训练样本。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from roboarena.tasks.api import Episode


@dataclass
class FrameStore:
    frames: np.ndarray  # [N, H, W, 3] uint8，全部回合首尾相接
    actions: np.ndarray  # [N, A] float32；每个回合的最后一帧没有动作，那一格填该回合的最后一个动作
    starts: np.ndarray  # [E] 每个回合第一帧的下标
    lengths: np.ndarray  # [E] 每个回合的动作数 T（帧数是 T + 1）

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode]) -> FrameStore:
        frames, actions, starts, lengths, offset = [], [], [], [], 0
        for ep in episodes:
            frames.append(ep.frames)
            actions.append(np.concatenate([ep.actions, ep.actions[-1:]]))
            starts.append(offset)
            lengths.append(len(ep))
            offset += len(ep) + 1
        if not frames:
            raise ValueError("没有可用的回合")
        return cls(
            np.concatenate(frames),
            np.concatenate(actions).astype(np.float32),
            np.array(starts, dtype=np.int64),
            np.array(lengths, dtype=np.int64),
        )

    @property
    def n_episodes(self) -> int:
        return len(self.starts)

    @property
    def n_samples(self) -> int:
        """有动作可学的帧数：每个回合的 T 帧（最后一帧之后没有动作）。"""
        return int(self.lengths.sum())

    def policy_batch(
        self, rng: np.random.Generator, batch: int, obs_frames: int, chunk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """随机取 batch 个时刻 t，返回 (画面 [B, obs_frames, H, W, 3]，动作 [B, chunk, A])。

        画面是 o_{t-obs_frames+1..t}，回合开头不够的用首帧补齐（评测时也是这样填历史的）。
        动作是 a_{t..t+chunk-1}，回合末尾不够的重复最后一个动作（推杆停在原地）。窗口不跨回合。
        """
        ep = rng.choice(self.n_episodes, size=batch, p=self.lengths / self.lengths.sum())
        t = rng.integers(0, self.lengths[ep])  # [B]，各回合内均匀
        start, last = self.starts[ep], self.lengths[ep] - 1
        obs_t = np.maximum(t[:, None] + np.arange(1 - obs_frames, 1)[None], 0)  # [B, obs_frames]
        act_t = np.minimum(t[:, None] + np.arange(chunk)[None], last[:, None])  # [B, chunk]
        return self.frames[start[:, None] + obs_t], self.actions[start[:, None] + act_t]


def random_shift(frames: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """把每个样本的画面整体平移随机的若干像素（同一样本的各帧平移量相同），露出的边用边缘像素填充。

    frames [B, K, H, W, 3]。这是像素输入的策略常用的数据增强，能明显减轻过拟合。
    """
    if max_shift <= 0:
        return frames
    b, k, h, w, _ = frames.shape
    padded = np.pad(
        frames, ((0, 0), (0, 0), (max_shift, max_shift), (max_shift, max_shift), (0, 0)), mode="edge"
    )
    dy = rng.integers(0, 2 * max_shift + 1, size=b)
    dx = rng.integers(0, 2 * max_shift + 1, size=b)
    rows = dy[:, None] + np.arange(h)[None]  # [B, H]
    cols = dx[:, None] + np.arange(w)[None]  # [B, W]
    return padded[
        np.arange(b)[:, None, None, None],
        np.arange(k)[None, :, None, None],
        rows[:, None, :, None],
        cols[:, None, None, :],
    ]
