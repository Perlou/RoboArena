"""不需要训练的策略。"""

from __future__ import annotations

import numpy as np

from roboarena.tasks.api import TaskSpec


class RandomPolicy:
    """在动作范围内均匀采样。策略库的下界。"""

    obs_frames = 1

    def __init__(self, spec: TaskSpec, chunk: int = 1, name: str = "random") -> None:
        self.name = name
        self.chunk = chunk
        self._low = spec.action_low
        self._high = spec.action_high

    def act(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shape = (frames.shape[0], self.chunk, self._low.shape[0])
        return rng.uniform(self._low, self._high, size=shape).astype(np.float32)
