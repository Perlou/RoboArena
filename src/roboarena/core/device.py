"""设备选择与冒烟模式开关。torch 在函数内导入，避免拖慢不需要它的命令。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def is_quick() -> bool:
    """环境变量 ROBOARENA_QUICK 置为非空且非 0 时，使用冒烟规模。"""
    return os.environ.get("ROBOARENA_QUICK", "") not in ("", "0")


def get_device(prefer: str | None = None) -> torch.device:
    """按 cuda → mps → cpu 选择设备；prefer 可强制指定（如测试里用 "cpu"）。"""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
