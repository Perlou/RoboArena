"""PushT 的真实仿真器：B 个互相独立的 gym-pusht 环境，逐个步进。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from roboarena.tasks.api import StepResult

if TYPE_CHECKING:
    from roboarena.tasks.pusht import PushTInit

SUCCESS_THRESHOLD = 0.95


def _make_env() -> Any:
    # 直接实例化环境类，不经过 gym.make：不要 TimeLimit 包装（步数上限由评测器掌握），
    # 也不要成功后继续步进时的警告（采集数据时需要成功之后的画面）。
    from gym_pusht.envs.pusht import PushTEnv

    return PushTEnv(obs_type="pixels_agent_pos", render_mode="rgb_array")


class PushTSim:
    name = "real"

    def __init__(self) -> None:
        self._envs: list[Any] = []
        self._n = 0

    def reset(self, inits: Sequence[PushTInit]) -> StepResult:
        self._n = len(inits)
        while len(self._envs) < self._n:
            self._envs.append(_make_env())
        obs = [
            env.reset(options={"reset_to_state": list(init.state)})[0]
            for env, init in zip(self._envs[: self._n], inits, strict=True)
        ]
        return self._pack(obs)

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self._n, 2):
            raise ValueError(f"动作的形状是 {actions.shape}，应为 [{self._n}, 2]")
        return self._pack([env.step(a)[0] for env, a in zip(self._envs[: self._n], actions, strict=True)])

    def _pack(self, obs: list[dict[str, np.ndarray]]) -> StepResult:
        envs = self._envs[: self._n]
        # 进度直接向环境要，不用 step 返回的 info：重置之后的那一帧没有 info["coverage"]
        progress = np.array([env._get_coverage() for env in envs], dtype=np.float32)
        poses = [(*env.block.position, env.block.angle) for env in envs]
        return StepResult(
            frames=np.stack([o["pixels"] for o in obs]).astype(np.uint8),
            progress=np.clip(progress, 0.0, 1.0),
            success=progress > SUCCESS_THRESHOLD,
            labels={
                "block_pose": np.array(poses, dtype=np.float32).reshape(self._n, 3),
                "agent_pos": np.array([env.agent.position for env in envs], dtype=np.float32).reshape(
                    self._n, 2
                ),
            },
        )

    def close(self) -> None:
        for env in self._envs:
            env.close()
        self._envs = []
