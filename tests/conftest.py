"""测试的公共部件：临时工作区、冒烟配置、一个只看画面就能解决玩具任务的策略。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from roboarena.config import Config, load_config
from roboarena.policies import register_policy_backend
from roboarena.store.workspace import Workspace
from roboarena.tasks.api import TaskSpec

REPO = Path(__file__).resolve().parents[1]
QUICK = REPO / "configs" / "quick.toml"


class ToySeekPolicy:
    """从画面里找出绿色目标方块的中心，把它当作动作。演示"外部策略实现接口并注册即可参加评测"。"""

    obs_frames = 1
    chunk = 1

    def __init__(self, name: str = "toy-seek") -> None:
        self.name = name

    def act(self, frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        last = frames[:, -1].astype(np.int32)  # [B, H, W, 3]
        r, g, b = last[..., 0], last[..., 1], last[..., 2]
        green = (g > 120) & (r < 80) & (b < 80)  # [B, H, W]
        height, width = green.shape[1:]
        actions = np.full((len(frames), 1, 2), 0.5, dtype=np.float32)
        for i, mask in enumerate(green):
            rows, cols = np.nonzero(mask)
            if len(rows):
                actions[i, 0] = ((cols.mean() + 0.5) / width, (rows.mean() + 0.5) / height)
        return actions


def _build_seek(
    spec: TaskSpec, params: Mapping[str, Any], checkpoint: Path | None, name: str
) -> ToySeekPolicy:
    return ToySeekPolicy(name)


register_policy_backend("toy_seek", _build_seek)


@pytest.fixture(autouse=True)
def _no_quick_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOARENA_QUICK", raising=False)
    monkeypatch.delenv("ROBOARENA_CONFIG_DIR", raising=False)


@pytest.fixture
def ws(tmp_path: Path) -> Iterator[Workspace]:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return load_config(QUICK, [f'workspace.root="{tmp_path / "ws"}"'])
