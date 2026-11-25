"""PushT 推块任务：用圆形推杆把 T 形块推到目标位置。真实仿真器是 gym-pusht（pymunk 二维物理）。

画面    96×96 RGB
动作    推杆的目标位置 [2]，像素坐标 0–512；环境内部用 PD 控制跟踪它
频率    10 赫兹，回合上限 300 步
进度    coverage：T 形块与目标区域的重叠比例
成功    coverage > 0.95
标签    block_pose [3]（块的 x、y、朝向）、agent_pos [2]（推杆位置）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboarena.tasks.api import Episode, Simulator, Task, TaskSpec


@dataclass(frozen=True)
class PushTInit:
    """交给 gym-pusht 的 reset_to_state：推杆 x、y，块 x、y，块朝向。

    注意：gym-pusht 设定朝向时会让块的位置发生偏移，所以这里的块坐标不等于重置后读到的 block_pose。
    它仍然唯一确定一个初始状态，这就够了。
    """

    state: tuple[float, float, float, float, float]


class PushTTask:
    spec = TaskSpec(
        name="pusht",
        frame_shape=(96, 96, 3),
        action_low=np.zeros(2, dtype=np.float32),
        action_high=np.full(2, 512.0, dtype=np.float32),
        control_hz=10.0,
        max_steps=300,
        label_fields={"block_pose": 3, "agent_pos": 2},
    )

    def sample_inits(self, n: int, seed: int) -> list[PushTInit]:
        """与 gym-pusht 自己的随机重置同分布：推杆 50–450，块 100–400，朝向 −π 到 π。"""
        rng = np.random.default_rng(seed)
        inits = []
        for _ in range(n):
            ax, ay = rng.integers(50, 450, size=2)
            bx, by = rng.integers(100, 400, size=2)
            inits.append(
                PushTInit((float(ax), float(ay), float(bx), float(by), float(rng.uniform(-np.pi, np.pi))))
            )
        return inits

    def make_real_sim(self) -> Simulator:
        from roboarena.tasks.pusht.sim import PushTSim

        return PushTSim()

    def encode_init(self, init: PushTInit) -> dict[str, Any]:
        return {"state": list(init.state)}

    def decode_init(self, data: dict[str, Any]) -> PushTInit:
        a, b, c, d, e = (float(v) for v in data["state"])
        return PushTInit((a, b, c, d, e))

    def demos(self, cache_dir: Path, limit: int | None = None) -> list[Episode]:
        from roboarena.tasks.pusht.demos import load_demos

        return load_demos(cache_dir / "pusht", limit)


def make_task() -> Task:
    return PushTTask()
