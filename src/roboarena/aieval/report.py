"""AI 评测的汇总与对比报告。全部由 ai_cases 的明细现算（docs/PLAN.md 1.2、1.3 节）。

比例：各种子的结果合并后算；另报种子之间最大与最小的差。
区间：95% Wilson 区间，样本量取用例数，不取「用例数 × 种子数」。统计单位是用例（PLAN 1.2）；同一用例的几个种子不是独立样本——
temperature 0 下它们常常逐字相同，按行数算会把样本量虚增、区间变窄。这等于把同一用例的各个种子当作完全相关，
种子之间确有差异时区间偏宽（保守）。
延迟：只算实际调用了模型的那些（命中缓存的不算）。
配对比较：以用例为单位。一个用例在某个配置下算「通过」，当它在严格多于一半的种子上通过；
然后在两个配置都跑过的用例上做 McNemar 精确检验。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from roboarena.aieval.cases import BUCKETS
from roboarena.eval.stats import mcnemar_exact, percentile, wilson_interval
from roboarena.store.aievals import ai_case_rows
from roboarena.store.workspace import Workspace

ALL = "all"
METRICS = ("valid", "match")
METRIC_ZH = {"valid": "合法率", "match": "完全匹配率"}


@dataclass(frozen=True)
class Rate:
    k: int  # 通过的（用例，种子）数
    n: int  # （用例，种子）总数
    n_cases: int  # 不同的用例数；区间按它算
    low: float
    high: float
    seed_spread: float  # 各种子的比例里最大与最小的差

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else 0.0


@dataclass(frozen=True)
class Summary:
    suite_fp: str
    model: str
    setup: str
    bucket: str  # seen / heldout / handwritten / all
    seeds: tuple[int, ...]
    n_cases: int
    valid: Rate
    match: Rate
    n_timed: int  # 实际调用了模型的次数
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    tokens_in_mean: float
    tokens_out_mean: float
    # 输出达到上限被截断的、被服务端中止的回应各有多少条（按用例 × 种子计）。早先的评测没记结束原因，为 None
    n_length: int | None
    n_aborted: int | None


@dataclass(frozen=True)
class Comparison:
    suite_fp: str
    setup_a: str
    setup_b: str
    model_a: str
    model_b: str
    bucket: str
    metric: str
    n_cases: int
    pass_a: int
    pass_b: int
    only_a: int
    only_b: int
    p_value: float


def _passed(row: dict[str, Any], metric: str) -> bool:
    return bool(json.loads(row["scores_json"])[metric])


def _rate(rows: Sequence[dict[str, Any]], metric: str) -> Rate:
    k = sum(_passed(r, metric) for r in rows)
    per_seed: dict[int, list[bool]] = {}
    for r in rows:
        per_seed.setdefault(r["seed"], []).append(_passed(r, metric))
    rates = [sum(v) / len(v) for v in per_seed.values()]
    n_cases = len({r["case_id"] for r in rows})
    pooled = k / len(rows) if rows else 0.0
    low, high = wilson_interval(pooled * n_cases, n_cases)  # 有效样本量 = 用例数
    return Rate(k, len(rows), n_cases, low, high, max(rates) - min(rates) if rates else 0.0)


def _in_bucket(rows: Sequence[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    return [r for r in rows if bucket == ALL or r["bucket"] == bucket]


def _grouped(ws: Workspace, suite: str) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ai_case_rows(ws, suite):
        groups.setdefault((row["suite_fp"], row["model"], row["setup"]), []).append(row)
    return groups


def summarize(ws: Workspace, suite: str) -> list[Summary]:
    """每个（用例集，模型，配置）在每个桶上的汇总。没有用例的桶不出现。"""
    out: list[Summary] = []
    for (fp, model, setup), rows in _grouped(ws, suite).items():
        for bucket in (*BUCKETS, ALL):
            chosen = _in_bucket(rows, bucket)
            if not chosen:
                continue
            scores = [json.loads(r["scores_json"]) for r in chosen]
            timed = [r["latency_ms"] for r, sc in zip(chosen, scores, strict=True) if not sc["cached"]]
            known = all("finish" in sc for sc in scores)
            out.append(
                Summary(
                    suite_fp=fp,
                    model=model,
                    setup=setup,
                    bucket=bucket,
                    seeds=tuple(sorted({r["seed"] for r in chosen})),
                    n_cases=len({r["case_id"] for r in chosen}),
                    valid=_rate(chosen, "valid"),
                    match=_rate(chosen, "match"),
                    n_timed=len(timed),
                    latency_p50_ms=percentile(timed, 50) if timed else None,
                    latency_p95_ms=percentile(timed, 95) if timed else None,
                    tokens_in_mean=sum(r["tokens_in"] for r in chosen) / len(chosen),
                    tokens_out_mean=sum(r["tokens_out"] for r in chosen) / len(chosen),
                    n_length=sum(sc["finish"] == "length" for sc in scores) if known else None,
                    n_aborted=sum(sc["finish"] == "aborted" for sc in scores) if known else None,
                )
            )
    return out


def _majority(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, bool]:
    """每个用例是否通过：在严格多于一半的种子上通过才算。"""
    votes: dict[str, list[bool]] = {}
    for r in rows:
        votes.setdefault(r["case_id"], []).append(_passed(r, metric))
    return {case: sum(v) * 2 > len(v) for case, v in votes.items()}


def compare(ws: Workspace, suite: str) -> list[Comparison]:
    """同一个用例集上的每一对（模型，配置），在每个桶、每个指标上的配对比较。"""
    out: list[Comparison] = []
    groups = _grouped(ws, suite)
    for (key_a, rows_a), (key_b, rows_b) in combinations(groups.items(), 2):
        if key_a[0] != key_b[0]:
            continue
        for bucket in (*BUCKETS, ALL):
            for metric in METRICS:
                got_a = _majority(_in_bucket(rows_a, bucket), metric)
                got_b = _majority(_in_bucket(rows_b, bucket), metric)
                shared = sorted(got_a.keys() & got_b.keys())
                if not shared:
                    continue
                only_a = sum(got_a[c] and not got_b[c] for c in shared)
                only_b = sum(got_b[c] and not got_a[c] for c in shared)
                out.append(
                    Comparison(
                        suite_fp=key_a[0],
                        setup_a=key_a[2],
                        setup_b=key_b[2],
                        model_a=key_a[1],
                        model_b=key_b[1],
                        bucket=bucket,
                        metric=metric,
                        n_cases=len(shared),
                        pass_a=sum(got_a[c] for c in shared),
                        pass_b=sum(got_b[c] for c in shared),
                        only_a=only_a,
                        only_b=only_b,
                        p_value=mcnemar_exact(only_a, only_b),
                    )
                )
    return out


def _fmt_rate(r: Rate) -> str:
    return f"{r.k}/{r.n} = {r.rate:.3f} [{r.low:.3f}, {r.high:.3f}]"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    return lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def render_report(ws: Workspace, suite: str) -> str:
    """Markdown 的对比报告。表里的数字都是从结果库现算的。"""
    summaries = summarize(ws, suite)
    if not summaries:
        return f"套件 {suite} 还没有评测结果。先运行 roboarena aieval run。\n"
    comparisons = compare(ws, suite)
    lines = [f"# AI 评测报告：{suite}", ""]
    for fp in dict.fromkeys(s.suite_fp for s in summaries):
        mine = [s for s in summaries if s.suite_fp == fp]
        n_all = next(s.n_cases for s in mine if s.bucket == ALL)
        lines += [f"## 用例集 {fp[:12]}（{n_all} 条用例）", ""]
        lines += [
            "比例是各种子合并后的值（分母 = 用例数 × 种子数）。方括号里是 95% Wilson 区间，按用例数算："
            "同一用例的几个种子不是独立样本。延迟不含命中缓存的调用。"
            "「截断 / 中止」是输出达到上限、被服务端中止的回应数（被中止的回应没有 token 数，平均值里按 0 算）。",
            "",
        ]
        rows = [
            (
                s.model, s.setup, s.bucket, len(s.seeds), s.n_cases, _fmt_rate(s.valid), _fmt_rate(s.match),
                f"{s.match.seed_spread:.3f}", f"{_ms(s.latency_p50_ms)} / {_ms(s.latency_p95_ms)}（{s.n_timed} 次）",
                f"{s.tokens_in_mean:.0f} / {s.tokens_out_mean:.0f}",
                "—" if s.n_length is None else f"{s.n_length} / {s.n_aborted}",
            )
            for s in mine
        ]  # fmt: skip
        lines += _md_table(
            ("模型", "配置", "桶", "种子数", "用例", "合法率", "完全匹配率", "种子间差（匹配）",
             "延迟 p50 / p95 毫秒", "平均 token 入 / 出", "截断 / 中止"),
            rows,
        )  # fmt: skip
        pairs = [c for c in comparisons if c.suite_fp == fp]
        if pairs:
            lines += ["", "### 配对比较（McNemar 精确检验，以用例为单位）", ""]
            lines += [
                "一个用例算通过：在严格多于一半的种子上通过。「只有甲」是甲通过而乙没通过的用例数。",
                "",
            ]
            rows = [
                (
                    f"{c.model_a}（{c.setup_a}）", f"{c.model_b}（{c.setup_b}）", c.bucket, METRIC_ZH[c.metric],
                    c.n_cases, c.pass_a, c.pass_b, c.only_a, c.only_b, f"{c.p_value:.4g}",
                )
                for c in pairs
            ]  # fmt: skip
            lines += _md_table(
                ("甲", "乙", "桶", "指标", "用例", "甲通过", "乙通过", "只有甲", "只有乙", "p"), rows
            )
        lines.append("")
    return "\n".join(lines)
