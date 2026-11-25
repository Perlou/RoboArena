"""回合文件：每个回合一个 .npz。字段见 roboarena.tasks.api.Episode。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboarena.core.hashing import digest_arrays
from roboarena.tasks.api import Episode

LABEL_PREFIX = "label__"


def _arrays(ep: Episode) -> dict[str, np.ndarray]:
    arrays = {"frames": ep.frames, "actions": ep.actions, "progress": ep.progress, "success": ep.success}
    arrays.update({LABEL_PREFIX + name: value for name, value in ep.labels.items()})
    return arrays


def episode_digest(ep: Episode) -> str:
    """回合内容的摘要。只算数组，不算 meta：同样的数据不管怎么描述，摘要都一样。"""
    return digest_arrays(_arrays(ep))


def save_episode(path: Path, ep: Episode) -> None:
    """先写临时文件再改名，中断时不会留下写了一半的文件。"""
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(ep.meta, ensure_ascii=False)), **_arrays(ep))  # type: ignore[arg-type]
    tmp.rename(path)


def load_episode(path: Path) -> Episode:
    with np.load(path, allow_pickle=False) as data:
        return Episode(
            frames=data["frames"],
            actions=data["actions"],
            progress=data["progress"],
            success=data["success"],
            labels={k[len(LABEL_PREFIX) :]: data[k] for k in data.files if k.startswith(LABEL_PREFIX)},
            meta=json.loads(str(data["meta"])),
        )
