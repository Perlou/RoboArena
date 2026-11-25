"""PushT 的人类演示：下载 Hugging Face 上的 lerobot/pusht（LeRobot v3.0 格式），解码成回合。

四个文件，约 8 MB：
    meta/info.json                                   元信息
    meta/episodes/chunk-000/file-000.parquet         每个回合在总表里的起止位置
    data/chunk-000/file-000.parquet                  逐帧的状态、动作、奖励
    videos/observation.image/chunk-000/file-000.mp4  全部回合首尾相接的 96×96 画面

演示里没有块的位姿，也没有回合的初始状态，所以不能在仿真器里按状态重放。
访问不了 huggingface.co 时可以设置镜像：HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from roboarena.core.errors import RoboArenaError
from roboarena.core.log import get_logger
from roboarena.tasks.api import Episode
from roboarena.tasks.pusht.sim import SUCCESS_THRESHOLD

log = get_logger(__name__)

FILES = [
    "meta/info.json",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
]


def download_raw(root: Path) -> Path:
    """把原始文件下载到 root/raw/，已存在的不重复下载。"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    raw = root / "raw"
    for name in FILES:
        target = raw / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/datasets/lerobot/pusht/resolve/main/{name}"
        log.info("下载演示数据", extra={"url": url})
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as err:
            tmp.unlink(missing_ok=True)
            raise RoboArenaError(
                f"下载 PushT 演示数据失败：{err}。可以设置镜像后重试：HF_ENDPOINT=https://hf-mirror.com"
            ) from err
        tmp.rename(target)
    return raw


def load_demos(root: Path, limit: int | None = None) -> list[Episode]:
    """返回前 limit 个演示回合（None 表示全部）。

    原始数据每个回合是 T 帧配 T 个动作，最后一个动作之后的画面不存在，所以丢掉最后一个动作。
    奖励是 clip(coverage / 0.95, 0, 1)：小于 1 时可以还原出 coverage，等于 1 时只知道 coverage ≥ 0.95，填 NaN。
    """
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    raw = download_raw(root)
    table = pq.read_table(raw / FILES[2]).to_pydict()
    index = pq.read_table(raw / FILES[1], columns=["dataset_from_index", "dataset_to_index"]).to_pydict()
    total = json.loads((raw / FILES[0]).read_text())["total_frames"]
    if table["index"] != list(range(total)):
        raise RoboArenaError("PushT 演示的逐帧表没有按全局下标排好序")

    starts, ends = index["dataset_from_index"], index["dataset_to_index"]
    n = len(starts) if limit is None else min(limit, len(starts))
    reader = imageio.get_reader(raw / FILES[3])
    images = np.stack([reader.get_data(i) for i in range(ends[n - 1])]).astype(np.uint8)  # 回合按顺序排列
    reader.close()

    states = np.asarray(table["observation.state"], dtype=np.float32)
    actions = np.asarray(table["action"], dtype=np.float32)
    rewards = np.asarray(table["next.reward"], dtype=np.float32)
    successes = np.asarray(table["next.success"], dtype=bool)

    episodes = []
    for i in range(n):
        s, e = starts[i], ends[i]
        length = e - s  # 帧数
        after = rewards[s : e - 1]  # 第 t 个动作之后的奖励，对应第 t + 1 帧
        progress = np.full(length, np.nan, dtype=np.float32)
        progress[1:] = np.where(after < 1.0, after * SUCCESS_THRESHOLD, np.nan)
        success = np.zeros(length, dtype=bool)
        success[1:] = successes[s : e - 1]
        episodes.append(
            Episode(
                frames=images[s:e],
                actions=actions[s : e - 1],
                progress=progress,
                success=success,
                labels={
                    "block_pose": np.full((length, 3), np.nan, dtype=np.float32),
                    "agent_pos": states[s:e],
                },
                meta={"source": "demo", "index": i, "init": None},
            )
        )
    return episodes
