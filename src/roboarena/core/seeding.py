"""随机种子。

每个组件用（全局种子，组件名）派生自己的种子，互不共享发生器。
这样在某个组件里多采一个随机数，不会改变其他组件的结果。
"""

from __future__ import annotations

import hashlib
import random

import numpy as np


def derive_seed(seed: int, name: str) -> int:
    """由全局种子和组件名得到一个 63 位的非负整数种子。相同输入永远得到相同输出。"""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def make_rng(seed: int, name: str) -> np.random.Generator:
    """为名为 name 的组件创建独立的 NumPy 发生器。"""
    return np.random.default_rng(derive_seed(seed, name))


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 的全局随机状态。只在训练入口调用一次；其余地方用 make_rng。"""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
