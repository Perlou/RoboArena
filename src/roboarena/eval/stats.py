"""评测用的统计量。每个函数都有对手算例子的测试。"""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """成功率的 Wilson 置信区间。

    比"均值 ± 1.96 个标准误"好在：成功数为 0 或等于总数时区间不会退化成一个点，也不会越出 [0, 1]。
    n_total 为 0 时返回 (0, 1)：没有数据，什么都不知道。
    """
    if n_total <= 0:
        return 0.0, 1.0
    if not 0 <= n_success <= n_total:
        raise ValueError(f"成功数 {n_success} 不在 0 到 {n_total} 之间")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = n_success / n_total
    denom = 1 + z * z / n_total
    center = (p + z * z / (2 * n_total)) / denom
    half = z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total * n_total)) / denom
    return max(0.0, center - half), min(1.0, center + half)
