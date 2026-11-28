"""AI 评测的汇总与对比报告。全部由 ai_cases 的明细现算（docs/PLAN.md 1.2、1.3 节）。

比例：各种子的结果合并后算；另报种子之间最大与最小的差。
区间：95% Wilson 区间，样本量取单个种子的统计单位数（用例数、每种子回答数、每种子数字数），不取「× 种子数」。
同一用例的几个种子不是独立样本——temperature 0 下它们常常逐字相同，按行数算会把样本量虚增、区间变窄。
延迟：只算实际调用了模型的那些（命中缓存的不算）。
配对比较：以用例为单位。一个用例在某个配置下算「通过」，当它在严格多于一半的种子上通过；
然后在两个配置都跑过的用例上做 McNemar 精确检验。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from roboarena.aieval.cases import BUCKETS
from roboarena.eval.stats import mcnemar_exact, percentile, wilson_interval
from roboarena.store.aievals import ai_case_rows
from roboarena.store.workspace import Workspace

ALL = "all"
METRICS = ("valid", "match")
TASK_METRICS = ("passed", "no_hallucination")
METRIC_ZH = {
    "valid": "合法率",
    "match": "完全匹配率",
    "passed": "任务完成率",
    "no_hallucination": "无幻觉（按用例）",
}


@dataclass(frozen=True)
class Rate:
    k: int  # 分子（合并全部种子）
    n: int  # 分母（合并全部种子）
    n_cases: int  # 有效样本量（单个种子的统计单位数）；区间按它算
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
class TaskSummary:
    suite_fp: str
    model: str
    setup: str
    seeds: tuple[int, ...]
    n_cases: int
    passed: Rate  # 任务完成率（按用例）
    hallucinated_answers: Rate  # 幻觉率（按回答）
    unsupported_numbers: Rate  # 幻觉率（按数字）
    violations: int  # 未经确认就执行了需确认操作的次数（目标是 0）
    confirm_attempts: int  # 发起需确认操作的次数
    retried: int  # 结果校验触发重答的轮次数
    n_calls_timed: int  # 未命中缓存的单次模型调用数
    call_p50_ms: float | None
    call_p95_ms: float | None
    n_cases_timed: int  # 全程未命中缓存的用例运行数
    case_p50_ms: float | None
    case_p95_ms: float | None
    steps_mean: float
    tokens_in_mean: float
    tokens_out_mean: float
    cost_mean: float
    n_length: int
    n_aborted: int


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
    if metric == "passed":
        return bool(row["passed"])
    sc = json.loads(row["scores_json"])
    if metric == "no_hallucination":
        return sc["hallucinated_answers"] == 0
    return bool(sc[metric])


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


def _ratio_rate(
    rows: Sequence[dict[str, Any]],
    scores: Sequence[dict[str, Any]],
    num_key: str,
    den_key: str,
) -> Rate:
    """统计单位是回答或数字时的比例（幻觉率）。有效样本量取每个种子的平均分母，不把种子数乘进去。"""
    per_seed: dict[int, list[tuple[int, int]]] = {}
    for r, sc in zip(rows, scores, strict=True):
        per_seed.setdefault(r["seed"], []).append((int(sc[num_key]), int(sc[den_key])))
    k = sum(sc[num_key] for sc in scores)
    n = sum(sc[den_key] for sc in scores)
    seed_rates = [
        sum(a for a, _ in pairs) / total if (total := sum(b for _, b in pairs)) else 0.0
        for pairs in per_seed.values()
    ]
    n_eff = max(1, round(n / len(per_seed))) if n > 0 and per_seed else 0
    pooled = k / n if n else 0.0
    low, high = wilson_interval(pooled * n_eff, n_eff)
    spread = max(seed_rates) - min(seed_rates) if seed_rates else 0.0
    return Rate(k, n, n_eff, low, high, spread)


def _in_bucket(rows: Sequence[dict[str, Any]], bucket: str) -> list[dict[str, Any]]:
    return [r for r in rows if bucket == ALL or r["bucket"] == bucket]


def _grouped(ws: Workspace, suite: str) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ai_case_rows(ws, suite):
        groups.setdefault((row["suite_fp"], row["model"], row["setup"]), []).append(row)
    return groups


def summarize(ws: Workspace, suite: str = "toolcall") -> list[Summary]:
    """toolcall 套件：每个（用例集，模型，配置）在每个桶上的汇总。没有用例的桶不出现。"""
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


def summarize_task(ws: Workspace) -> list[TaskSummary]:
    """task 套件：每个（用例集，模型，配置）的汇总。"""
    out: list[TaskSummary] = []
    for (fp, model, setup), rows in _grouped(ws, "task").items():
        scores = [json.loads(r["scores_json"]) for r in rows]
        call_ms = [float(ms) for sc in scores for ms in sc.get("call_ms", [])]
        case_ms = [float(r["latency_ms"]) for r, sc in zip(rows, scores, strict=True) if not sc["cached"]]
        out.append(
            TaskSummary(
                suite_fp=fp,
                model=model,
                setup=setup,
                seeds=tuple(sorted({r["seed"] for r in rows})),
                n_cases=len({r["case_id"] for r in rows}),
                passed=_rate(rows, "passed"),
                hallucinated_answers=_ratio_rate(rows, scores, "hallucinated_answers", "answers"),
                unsupported_numbers=_ratio_rate(rows, scores, "unsupported_numbers", "numbers"),
                violations=sum(int(sc["violations"]) for sc in scores),
                confirm_attempts=sum(int(sc["confirm_attempts"]) for sc in scores),
                retried=sum(int(sc.get("retried", 0)) for sc in scores),
                n_calls_timed=len(call_ms),
                call_p50_ms=percentile(call_ms, 50) if call_ms else None,
                call_p95_ms=percentile(call_ms, 95) if call_ms else None,
                n_cases_timed=len(case_ms),
                case_p50_ms=percentile(case_ms, 50) if case_ms else None,
                case_p95_ms=percentile(case_ms, 95) if case_ms else None,
                steps_mean=sum(r["steps"] for r in rows) / len(rows),
                tokens_in_mean=sum(r["tokens_in"] for r in rows) / len(rows),
                tokens_out_mean=sum(r["tokens_out"] for r in rows) / len(rows),
                cost_mean=sum(float(sc.get("cost", 0.0)) for sc in scores) / len(scores),
                n_length=sum(int(sc.get("n_length", 0)) for sc in scores),
                n_aborted=sum(int(sc.get("n_aborted", 0)) for sc in scores),
            )
        )
    return out


def _majority(rows: Sequence[dict[str, Any]], check: Callable[[dict[str, Any]], bool]) -> dict[str, bool]:
    """每个用例是否通过：在严格多于一半的种子上通过才算。"""
    votes: dict[str, list[bool]] = {}
    for r in rows:
        votes.setdefault(r["case_id"], []).append(check(r))
    return {case: sum(v) * 2 > len(v) for case, v in votes.items()}


def compare(ws: Workspace, suite: str) -> list[Comparison]:
    """同一个用例集上的每一对（模型，配置），在每个桶、每个指标上的配对比较。"""
    out: list[Comparison] = []
    groups = _grouped(ws, suite)
    buckets = (ALL,) if suite == "task" else (*BUCKETS, ALL)
    metrics = TASK_METRICS if suite == "task" else METRICS
    for (key_a, rows_a), (key_b, rows_b) in combinations(groups.items(), 2):
        if key_a[0] != key_b[0]:
            continue
        for bucket in buckets:
            for metric in metrics:
                got_a = _majority(_in_bucket(rows_a, bucket), lambda r, m=metric: _passed(r, m))
                got_b = _majority(_in_bucket(rows_b, bucket), lambda r, m=metric: _passed(r, m))
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


def _render_task_report(ws: Workspace) -> str:
    summaries = summarize_task(ws)
    if not summaries:
        return "套件 task 还没有评测结果。先运行 roboarena aieval run --suite task。\n"
    comparisons = compare(ws, "task")
    lines = ["# AI 评测报告：task", ""]
    for fp in dict.fromkeys(s.suite_fp for s in summaries):
        mine = [s for s in summaries if s.suite_fp == fp]
        lines += [f"## 用例集 {fp[:12]}（{mine[0].n_cases} 条用例）", ""]
        lines += [
            "比例是各种子合并后的值；95% Wilson 区间按单个种子的统计单位数算（用例数、每种子回答数、每种子数字数），"
            "不把重复种子算作独立样本。延迟不含命中缓存的调用。",
            "",
        ]
        cap_rows = [
            (
                s.model, s.setup, len(s.seeds), s.n_cases, _fmt_rate(s.passed), f"{s.passed.seed_spread:.3f}",
                _fmt_rate(s.hallucinated_answers), _fmt_rate(s.unsupported_numbers),
                f"{s.violations} / {s.confirm_attempts}", s.retried,
            )
            for s in mine
        ]  # fmt: skip
        lines += _md_table(
            ("模型", "配置", "种子数", "用例", "任务完成率", "种子间差",
             "幻觉率（按回答）", "幻觉率（按数字）", "越权 / 需确认", "校验重答"),
            cap_rows,
        )  # fmt: skip
        lines.append("")
        perf_rows = [
            (
                s.model, s.setup,
                f"{_ms(s.call_p50_ms)} / {_ms(s.call_p95_ms)}（{s.n_calls_timed} 次）",
                f"{_ms(s.case_p50_ms)} / {_ms(s.case_p95_ms)}（{s.n_cases_timed} 例）",
                f"{s.steps_mean:.1f}", f"{s.tokens_in_mean:.0f} / {s.tokens_out_mean:.0f}",
                f"{s.cost_mean:.4g}", f"{s.n_length} / {s.n_aborted}",
            )
            for s in mine
        ]  # fmt: skip
        lines += _md_table(
            ("模型", "配置", "单次调用 p50 / p95 毫秒", "用例总耗时 p50 / p95 毫秒",
             "平均步数", "平均 token 入 / 出", "平均成本", "截断 / 中止"),
            perf_rows,
        )  # fmt: skip
        pairs = [c for c in comparisons if c.suite_fp == fp]
        if pairs:
            lines += ["", "### 配对比较（McNemar 精确检验，以用例为单位）", ""]
            lines += [
                "一个用例算通过：在严格多于一半的种子上通过。「只有甲」是甲通过而乙没通过的用例数。",
                "",
            ]
            pair_rows = [
                (
                    f"{c.model_a}（{c.setup_a}）", f"{c.model_b}（{c.setup_b}）", METRIC_ZH[c.metric],
                    c.n_cases, c.pass_a, c.pass_b, c.only_a, c.only_b, f"{c.p_value:.4g}",
                )
                for c in pairs
            ]  # fmt: skip
            lines += _md_table(
                ("甲", "乙", "指标", "用例", "甲通过", "乙通过", "只有甲", "只有乙", "p"), pair_rows
            )
        lines.append("")
    return "\n".join(lines)


def render_report(ws: Workspace, suite: str) -> str:
    """Markdown 的对比报告。表里的数字都是从结果库现算的。"""
    if suite == "task":
        return _render_task_report(ws)
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
