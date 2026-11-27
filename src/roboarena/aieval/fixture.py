"""固定工作区：AI 评测的全部用例都在它上面运行（docs/PLAN.md 1.1 节）。

玩具任务、冒烟规模，在 CPU 上几秒建好，不联网。里面的名称（策略、评测集、数据集）是专门起的一套，
微调语料用另一套（A3），防止模型把名字背下来。真值由平台函数在这个工作区上现算。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from roboarena.config import Config, config_dir, load_config, with_overrides
from roboarena.core.hashing import sha256_text
from roboarena.core.log import get_logger
from roboarena.data.dataset import build_dataset
from roboarena.eval import service
from roboarena.policies.zoo import build_zoo, zoo_names
from roboarena.store.workspace import Workspace

log = get_logger(__name__)

BASE = "atlas"
SUITES = ("bench", "probe")  # 主评测集；另一个小的，只评了一部分策略
DATASETS = ("core", "mini")
PROBE_POLICIES = (BASE, f"{BASE}-noise0.3", "random")  # 在 probe 上评过的策略

_OVERRIDES = (
    f'zoo.base="{BASE}"',
    "zoo.diffusion=false",
    "zoo.exec_noise=[0.1, 0.3]",
    "policy.steps=200",
    "policy.snapshots=[20, 60]",
    f'eval.suite="{SUITES[0]}"',
    f'data.name="{DATASETS[0]}"',
)
_PROBE = (f'eval.suite="{SUITES[1]}"', "eval.n_inits=8")
_MINI = (f'data.name="{DATASETS[1]}"', "data.n_demos=4", "data.n_replay=2", "data.n_random_walk=2")
_VERSION = 1  # 固定工作区的内容有变时加一，旧的缓存就不再使用


def fixture_config(root: Path) -> Config:
    """指向 root 的固定工作区配置。root 可以是建好的工作区，也可以是它的一份副本。"""
    return load_config(config_dir() / "quick.toml", [f'workspace.root="{root.as_posix()}"', *_OVERRIDES])


POLICIES: tuple[str, ...] = tuple(zoo_names(fixture_config(Path("unused"))))


@contextmanager
def _on_cpu() -> Iterator[None]:
    """固定工作区一律在 CPU 上建：结果逐位确定，与机器无关。"""
    previous = os.environ.get("ROBOARENA_DEVICE")
    os.environ["ROBOARENA_DEVICE"] = "cpu"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["ROBOARENA_DEVICE"]
        else:
            os.environ["ROBOARENA_DEVICE"] = previous


def build_fixture(root: Path) -> Config:
    """在 root 建固定工作区：两个数据集、两个评测集、七个策略和它们的评测结果。"""
    cfg = fixture_config(root)
    probe = with_overrides(cfg, _PROBE)
    with _on_cpu(), Workspace(root) as ws:
        build_dataset(ws, cfg)
        build_dataset(ws, with_overrides(cfg, _MINI))
        service.create_suite(ws, cfg)
        service.create_suite(ws, probe)
        names, _ = build_zoo(ws, cfg)
        for name in names:
            service.run_eval(ws, cfg, name)
        for name in PROBE_POLICIES:
            service.run_eval(ws, probe, name)
    return cfg


def ensure_fixture(cache: Path) -> Path:
    """返回建好的固定工作区的位置；cache 下没有时现建。建到一半中断不会留下半成品。"""
    digest = sha256_text(f"{_VERSION}|{fixture_config(Path('unused')).to_toml()}")[:12]
    root = cache / f"fixture-{digest}"
    if not root.exists():
        tmp = cache / f"fixture-{digest}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        build_fixture(tmp)
        tmp.rename(root)
        log.info("已建好固定工作区", extra={"root": str(root)})
    return root


def copy_fixture(src: Path, dst: Path) -> Config:
    """复制一份固定工作区，给会改动工作区的用例用。数据库用 backup 接口复制（WAL 模式下直接复制文件不可靠）。"""
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("roboarena.db*"))
    source = sqlite3.connect(src / "roboarena.db")
    target = sqlite3.connect(dst / "roboarena.db")
    try:
        source.backup(target)
    finally:
        source.close()
        target.close()
    return fixture_config(dst)
