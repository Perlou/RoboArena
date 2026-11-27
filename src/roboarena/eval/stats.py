"""评测用的统计量。每个函数都有对手算例子的测试。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist


def wilson_interval(n_success: float, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """成功率的 Wilson 置信区间。

    n_success 可以不是整数：重复测量折算成有效样本量之后（比例 × 有效样本数）会出现小数。

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


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """McNemar 精确检验的双侧 p 值：同一组用例上比较两个配置。

    only_a 是「甲对乙错」的用例数，only_b 是「乙对甲错」的用例数；两边都对或都错的用例不带信息，不参与。
    原假设下，n = only_a + only_b 个不一致的用例各以 1/2 的概率落在任一边，所以
    p = min(1, 2 · P(X ≤ min(only_a, only_b)))，X ~ Binomial(n, 1/2)。没有不一致的用例时 p = 1。
    """
    if only_a < 0 or only_b < 0:
        raise ValueError(f"不一致的用例数不能是负数：{only_a}、{only_b}")
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(values: Sequence[float], q: float) -> float:
    """分位数，最近秩法：排序后第 ceil(q/100 · n) 个值（从 1 数）。q 在 0 到 100 之间；q = 0 取最小值。

    不做插值：报告的延迟总是某一次实际测到的值。空序列报错。
    """
    if not values:
        raise ValueError("空序列没有分位数")
    if not 0 <= q <= 100:
        raise ValueError(f"分位数应在 0 到 100 之间：{q}")
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q / 100 * len(ordered))) - 1]
