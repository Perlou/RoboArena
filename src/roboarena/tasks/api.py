"""任务与仿真器的接口。评测器、服务、前端只认这里的类型，不认识具体任务。

形状约定：B 并行回合数，A 动作维度，H × W 画面尺寸（都由任务给出）。
画面在接口边界上一律是 uint8、取值 0–255；动作一律是环境坐标。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# 初始状态对评测器是不透明的：只从评测集里取出来，原样交给仿真器。
InitState = Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    frame_shape: tuple[int, int, int]  # (H, W, 3)
    action_low: np.ndarray  # [A]
    action_high: np.ndarray  # [A]
    control_hz: float
    max_steps: int
    label_fields: dict[str, int] = field(default_factory=dict)  # 判定头要回归的量及其维度

    @property
    def action_dim(self) -> int:
        return int(self.action_low.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frame_shape": list(self.frame_shape),
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "control_hz": self.control_hz,
            "max_steps": self.max_steps,
            "label_fields": dict(self.label_fields),
        }


@dataclass
class StepResult:
    frames: np.ndarray  # [B, H, W, 3] uint8
    progress: np.ndarray  # [B] float32，0–1 的任务进度
    success: np.ndarray  # [B] bool
    labels: dict[str, np.ndarray] = field(default_factory=dict)  # 任务声明的标签，各 [B, D]


class Simulator(Protocol):
    name: str

    def reset(self, inits: Sequence[InitState]) -> np.ndarray:
        """把 B = len(inits) 个回合设到给定的初始状态，返回首帧 [B, H, W, 3]。"""
        ...

    def step(self, actions: np.ndarray) -> StepResult:
        """actions [B, A]，环境坐标。"""
        ...

    def close(self) -> None: ...


class Task(Protocol):
    spec: TaskSpec

    def sample_inits(self, n: int, seed: int) -> list[InitState]:
        """采样 n 个初始状态，由 seed 完全确定。"""
        ...

    def make_real_sim(self) -> Simulator:
        """构造真实仿真器。回合数在 reset 时由 inits 的个数决定。"""
        ...

    def encode_init(self, init: InitState) -> dict[str, Any]:
        """序列化成可以写进 JSON 的字典。"""
        ...

    def decode_init(self, data: dict[str, Any]) -> InitState: ...
