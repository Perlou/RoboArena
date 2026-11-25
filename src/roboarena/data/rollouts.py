"""策略回合集：一组已登记的策略在真实仿真器里跑出来的回合，存成回合文件，供数据集并入。

    rollouts/<任务>/<名称>/
        episode-000000.npz …
        index.json        每个回合来自哪个策略、多长、是否成功

回合由上层（eval）采集后交给这里保存：数据层不认识策略。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from roboarena.core.errors import NotFoundError
from roboarena.data.episodes import load_episode, save_episode
from roboarena.tasks.api import Episode


def rollout_dir(root: Path, task: str, name: str) -> Path:
    return root / "rollouts" / task / name


def save_rollout_set(root: Path, task: str, name: str, episodes: Sequence[Episode]) -> Path:
    """保存一个回合集，覆盖同名的旧集合。root 是工作区的根目录。"""
    out = rollout_dir(root, task, name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index: list[dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        file = f"episode-{i:06d}.npz"
        save_episode(out / file, ep)
        index.append(
            {
                "file": file,
                "source": ep.meta["source"],
                "length": len(ep),
                "succeeded": bool(ep.success.any()),
            }
        )
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def rollout_index(root: Path, task: str, name: str) -> list[dict[str, Any]]:
    path = rollout_dir(root, task, name) / "index.json"
    if not path.exists():
        raise NotFoundError(f"任务 {task} 没有名为 {name} 的策略回合集，先运行 roboarena data rollouts")
    return json.loads(path.read_text(encoding="utf-8"))


def load_rollout_set(root: Path, task: str, name: str) -> list[Episode]:
    out = rollout_dir(root, task, name)
    return [load_episode(out / entry["file"]) for entry in rollout_index(root, task, name)]
