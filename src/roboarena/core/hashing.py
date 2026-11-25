"""内容哈希：文件、JSON、数组。数据集指纹和检查点的完整性校验都建立在这里。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_json(obj: Any) -> str:
    """键排序、无多余空白的 JSON。同样的内容总是得到同样的字符串，可以拿来求哈希。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    """对一组命名数组的内容求摘要：名称、类型、形状、原始字节都计入。

    不直接对 .npz 文件求哈希，因为压缩包里带时间戳，内容不变哈希也会变。
    """
    h = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def combine_digests(digests: Iterable[str]) -> str:
    """把多个摘要合成一个，与顺序无关。"""
    return sha256_text("\n".join(sorted(digests)))
