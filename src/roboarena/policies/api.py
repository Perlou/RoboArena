"""策略接口。策略只接收画面，这样同一个策略可以不加改动地接真实仿真器和神经仿真器。"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str
    obs_frames: int  # 需要的历史帧数
    chunk: int  # 一次输出的动作个数（动作分块）

    def act(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """frames [B, obs_frames, H, W, 3] uint8 → 动作 [B, chunk, A]，环境坐标。"""
        ...
