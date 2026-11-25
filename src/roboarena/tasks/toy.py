"""玩具任务：二维平面上把一个点移到目标点。

只用于测试和持续集成：不依赖任何物理引擎，几毫秒跑完一个回合。
它也是任务接口的第二个使用者，用来检验这套抽象没有夹带某个真实任务的假设。

    状态    点的位置 [2]、目标的位置 [2]，都在 [0, 1]
    动作    点要去的位置 [2]；每步最多移动 MAX_MOVE
    画面    32×32：白底，绿色方块是目标，红色方块是点
    进度    1 − 到目标的距离（截断到 0–1）
    成功    距离小于 SUCCESS_RADIUS
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from roboarena.tasks.api import Simulator, StepResult, Task, TaskSpec

SIZE = 32
MAX_MOVE = 0.15
SUCCESS_RADIUS = 0.1
MARGIN = 0.1  # 采样时离边界的距离，保证方块画得完整


@dataclass(frozen=True)
class ToyInit:
    agent: tuple[float, float]
    goal: tuple[float, float]


def _draw(canvas: np.ndarray, pos: np.ndarray, half: int, color: tuple[int, int, int]) -> None:
    """在 canvas [H, W, 3] 上以 pos（x 向右，y 向下，0–1）为中心画一个边长 2·half+1 的方块。"""
    col, row = (np.clip(pos, 0.0, 1.0 - 1e-6) * SIZE).astype(int)
    canvas[max(row - half, 0) : row + half + 1, max(col - half, 0) : col + half + 1] = color


def render(agent: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """agent、goal 都是 [B, 2]，返回 [B, 32, 32, 3] uint8。"""
    frames = np.full((len(agent), SIZE, SIZE, 3), 255, dtype=np.uint8)
    for i in range(len(agent)):
        _draw(frames[i], goal[i], 2, (0, 170, 0))
        _draw(frames[i], agent[i], 1, (220, 0, 0))
    return frames


class ToySim:
    name = "real"

    def __init__(self) -> None:
        self.agent = np.zeros((0, 2), dtype=np.float32)
        self.goal = np.zeros((0, 2), dtype=np.float32)

    def reset(self, inits: Sequence[ToyInit]) -> np.ndarray:
        self.agent = np.array([i.agent for i in inits], dtype=np.float32)
        self.goal = np.array([i.goal for i in inits], dtype=np.float32)
        return render(self.agent, self.goal)

    def step(self, actions: np.ndarray) -> StepResult:
        target = np.clip(np.asarray(actions, dtype=np.float32), 0.0, 1.0)
        delta = target - self.agent  # [B, 2]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, MAX_MOVE / np.maximum(length, 1e-8))
        self.agent = (self.agent + delta * scale).astype(np.float32)
        dist = np.linalg.norm(self.agent - self.goal, axis=1)
        return StepResult(
            frames=render(self.agent, self.goal),
            progress=np.clip(1.0 - dist, 0.0, 1.0).astype(np.float32),
            success=dist < SUCCESS_RADIUS,
            labels={"agent_pos": self.agent.copy()},
        )

    def close(self) -> None:
        pass


class ToyTask:
    spec = TaskSpec(
        name="toy",
        frame_shape=(SIZE, SIZE, 3),
        action_low=np.zeros(2, dtype=np.float32),
        action_high=np.ones(2, dtype=np.float32),
        control_hz=10.0,
        max_steps=30,
        label_fields={"agent_pos": 2},
    )

    def sample_inits(self, n: int, seed: int) -> list[ToyInit]:
        rng = np.random.default_rng(seed)
        points = rng.uniform(MARGIN, 1.0 - MARGIN, size=(n, 2, 2))
        return [ToyInit(agent=(float(a[0]), float(a[1])), goal=(float(g[0]), float(g[1]))) for a, g in points]

    def make_real_sim(self) -> Simulator:
        return ToySim()

    def encode_init(self, init: ToyInit) -> dict[str, Any]:
        return {"agent": list(init.agent), "goal": list(init.goal)}

    def decode_init(self, data: dict[str, Any]) -> ToyInit:
        ax, ay = data["agent"]
        gx, gy = data["goal"]
        return ToyInit(agent=(float(ax), float(ay)), goal=(float(gx), float(gy)))


def make_task() -> Task:
    return ToyTask()
