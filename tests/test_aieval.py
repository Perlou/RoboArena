"""AI 评测：用例集、检查器（正确的判过，故意做错的判不过）、运行与入库、汇总与配对比较。用脚本化假模型。"""

import json
from collections import Counter

import pytest

from roboarena.agent.tools import TOOLS, tool_specs
from roboarena.aieval import fixture
from roboarena.aieval.cases import FAMILIES, Case, suite_fingerprint, toolcall_cases
from roboarena.aieval.check import check_toolcall
from roboarena.aieval.report import ALL, compare, render_report, summarize
from roboarena.aieval.run import SYSTEM_PROMPT, run_toolcall, setup_id
from roboarena.config import LLMConfig, LLMModelConfig
from roboarena.core.errors import ModelError
from roboarena.eval.stats import percentile, wilson_interval
from roboarena.llm.api import Completion, ToolCall
from roboarena.llm.gateway import Gateway, read_calls
from roboarena.llm.scripted import ScriptedModel
from roboarena.store.registry import list_artifacts
from roboarena.store.workspace import Workspace

# ---- 用例集 ----


def test_suite_meets_the_planned_sizes_and_is_deterministic():
    cases = toolcall_cases()
    sizes = Counter(c.bucket for c in cases)
    assert sizes["seen"] >= 100 and sizes["heldout"] >= 100 and sizes["handwritten"] >= 60
    assert len({c.id for c in cases}) == len(cases) == len({c.turns for c in cases})
    assert suite_fingerprint(cases) == suite_fingerprint(list(reversed(toolcall_cases())))
    assert suite_fingerprint(cases) != suite_fingerprint(toolcall_cases(seed=1))
    assert suite_fingerprint(cases) != suite_fingerprint(cases[:-1])


def test_every_gold_call_is_valid_and_matches_itself():
    for case in toolcall_cases():
        assert case.gold_calls is not None and len(case.gold_calls) == 1
        assert case.gold_calls[0].name in TOOLS
        assert check_toolcall(case, case.gold_calls).match, case.id


def test_heldout_templates_and_families_stay_out_of_seen():
    cases = toolcall_cases()
    seen_groups = {c.group for c in cases if c.bucket == "seen"}
    assert not seen_groups & {c.group for c in cases if c.bucket != "seen"}  # 切分以组为单位
    whole = {f.name for f in FAMILIES if not f.seen}
    assert whole == {"leaderboard_top", "remove_policy"}
    assert not [c for c in cases if c.bucket == "seen" and c.family in whole]
    assert "remove_policy" not in {c.gold_calls[0].name for c in cases if c.bucket == "seen" and c.gold_calls}
    for family in FAMILIES:
        assert not set(family.seen) & set(family.heldout)


def test_cases_use_names_that_exist_in_the_fixture(fixture_root):
    with Workspace(fixture_root) as ws:
        policies = {a.name for a in list_artifacts(ws)}
        suites = {r[0] for r in ws.db.execute("SELECT name FROM suites")}
    assert policies == set(fixture.POLICIES) and suites == set(fixture.SUITES)
    assert {p.name for p in (fixture_root / "datasets" / "toy").iterdir()} == set(fixture.DATASETS)
    for case in toolcall_cases():
        assert case.gold_calls is not None
        call = case.gold_calls[0]
        args = call.arguments
        if call.name in ("compare_policies", "run_eval", "remove_policy"):
            assert {args[k] for k in ("a", "b", "policy") if k in args} <= policies, case.id
            if call.name == "remove_policy":
                assert args["name"] in policies, case.id
        if args.get("suite"):
            assert args["suite"] in suites, case.id
        if call.name == "dataset_info" and args.get("name"):
            assert args["name"] in fixture.DATASETS, case.id


# ---- 检查器 ----


def _case(tool: str, args: dict) -> Case:
    return Case("c", "f", "seen", "g", ("…",), (ToolCall("gold", tool, args),), "toolcall")


EVAL = _case("run_eval", {"policy": "atlas", "suite": "bench"})
TRAIN = _case(
    "train_policy", {"name": "nova", "backend": "bc", "overrides": ["policy.steps=500", "policy.lr=1e-3"]}
)
COMPARE = _case("compare_policies", {"a": "atlas", "b": "random"})


@pytest.mark.parametrize(
    "case, name, args",
    [
        (EVAL, "run_eval", {"policy": "atlas", "suite": "bench"}),
        (EVAL, "run_eval", {"suite": " bench ", "policy": "atlas", "force": False}),  # 默认值写出来、多了空白
        (EVAL, "run_eval", {"policy": "atlas", "suite": "bench", "force": None}),
        (
            TRAIN,
            "train_policy",
            {"name": "nova", "overrides": ["policy.lr=0.001", "policy.steps = 500"]},
        ),  # 顺序、写法
        (
            TRAIN,
            "train_policy",
            {"name": "nova", "backend": "bc", "overrides": ["policy.steps=500.0", "policy.lr=1e-3"]},
        ),
        (COMPARE, "compare_policies", {"a": "random", "b": "atlas"}),  # 两个策略互换，意思不变
        (_case("leaderboard", {"top": 3}), "leaderboard", {"top": 3.0, "suite": ""}),
    ],
)
def test_correct_calls_pass(case, name, args):
    score = check_toolcall(case, [ToolCall("x", name, args)])
    assert (score.valid, score.match, score.reason) == (True, True, "")


@pytest.mark.parametrize(
    "case, calls, valid, reason",
    [
        (EVAL, [], False, "没有工具调用"),
        (
            EVAL,
            [ToolCall("x", "leaderboard", {"suite": "bench"})],
            True,
            "工具不对：应为 run_eval，实际 leaderboard",
        ),
        (EVAL, [ToolCall("x", "evaluate", {"policy": "atlas"})], False, "工具不存在：evaluate"),
        (EVAL, [ToolCall("x", "run_eval", {"policy": "atlas"})], True, "suite 应为 'bench'，实际 ''"),
        (
            EVAL,
            [ToolCall("x", "run_eval", {"policy": "Atlas", "suite": "bench"})],
            True,
            "policy 应为 'atlas'",
        ),
        (
            EVAL,
            [ToolCall("x", "run_eval", {"policy": "atlas", "suite": "bench", "force": True})],
            True,
            "force 应为 False",
        ),
        (
            EVAL,
            [ToolCall("x", "run_eval", {"policy": "atlas", "suite": "bench", "sim": "real"})],
            False,
            "未知的参数：sim",
        ),
        (
            EVAL,
            [ToolCall("x", "run_eval", {"policy": "atlas", "force": "true"})],
            False,
            "force 应为 true 或 false",
        ),
        (
            EVAL,
            [ToolCall("x", "run_eval", {}, raw_arguments='{"policy": ')],
            False,
            "参数不是合法的 JSON 对象",
        ),
        (
            EVAL,
            [ToolCall("x", "run_eval", {"policy": "atlas", "suite": "bench"})] * 2,
            True,
            "应调用 1 次工具，实际 2 次",
        ),
        (
            TRAIN,
            [ToolCall("x", "train_policy", {"name": "nova", "overrides": ["policy.steps=500"]})],
            True,
            "overrides 应为",
        ),
        (
            TRAIN,
            [
                ToolCall(
                    "x",
                    "train_policy",
                    {"name": "nova", "overrides": ["policy.steps=500", "policy.lr=1e-3", "policy.batch=8"]},
                )
            ],
            True,
            "overrides 应为",
        ),
        (
            TRAIN,
            [
                ToolCall(
                    "x", "train_policy", {"name": "nova", "overrides": ["policy.steps=500", "policy.lr=1e-2"]}
                )
            ],
            True,
            "overrides",
        ),
        (
            TRAIN,
            [
                ToolCall(
                    "x",
                    "train_policy",
                    {"name": "nova", "overrides": ["diffusion.steps=500", "diffusion.lr=1e-3"]},
                )
            ],
            True,
            "overrides",
        ),
        (
            COMPARE,
            [ToolCall("x", "compare_policies", {"a": "atlas", "b": "atlas-step20"})],
            True,
            "参数不一致",
        ),
        (
            _case("leaderboard", {"top": 3}),
            [ToolCall("x", "leaderboard", {"top": "3"})],
            False,
            "top 应为整数",
        ),
    ],
)
def test_wrong_calls_fail(case, calls, valid, reason):
    score = check_toolcall(case, calls)
    assert not score.match and score.valid is valid and reason in score.reason


def test_checker_refuses_a_case_without_gold_calls():
    with pytest.raises(ValueError, match="没有标准工具调用"):
        check_toolcall(Case("c", "f", "seen", "g", ("…",), None, "task"), [])


# ---- 分位数：对手算例子 ----


def test_percentile_hand_computed():
    values = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert percentile(values, 50) == 30.0  # ceil(0.5 · 5) = 3，第 3 小
    assert percentile(values, 95) == 50.0  # ceil(4.75) = 5
    assert percentile(values, 20) == 10.0 and percentile(values, 21) == 20.0
    assert percentile(values, 0) == 10.0 and percentile(values, 100) == 50.0
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile(values, 101)


# ---- 运行与入库：假模型的得分可以预先算出 ----


def _sample(bucket: str) -> list[Case]:
    """这个桶里等间隔抽 6 条。标准答案是 list_policies 的不要：下面拿它当「故意答错」的回答。"""
    pool = [
        c
        for c in toolcall_cases()
        if c.bucket == bucket and c.gold_calls and c.gold_calls[0].name != "list_policies"
    ]
    return pool[:: len(pool) // 6][:6]


CASES = [*_sample("seen"), *_sample("heldout")]  # 12 条：位置 0–5 是 seen，6–11 是 heldout
WRONG = ToolCall("w", "list_policies", {})


def _gateway(tmp_path, replies_by_model: dict[str, list], **cfg) -> Gateway:
    llm = LLMConfig(
        default="good", models={name: LLMModelConfig(model=name) for name in replies_by_model}, **cfg
    )
    models = {name: ScriptedModel(name, replies) for name, replies in replies_by_model.items()}
    return Gateway(llm, cache_dir=tmp_path / "llm-cache", models=models)


def _gold(cases, wrong_at=()) -> list:
    """照标准答案作答，wrong_at 里的位置故意答错。"""
    return [WRONG if i in wrong_at else c.gold_calls[0] for i, c in enumerate(cases)]


def test_split_of_the_sample_is_as_assumed():
    assert [c.bucket for c in CASES] == ["seen"] * 6 + ["heldout"] * 6
    assert len({c.id for c in CASES}) == 12


def test_scripted_model_scores_exactly_what_was_scripted(ws, cfg, tmp_path):
    # 位置 1、2（seen）与 7（heldout）答错；位置 3 一个字不说；位置 8 的参数不是 JSON
    replies = _gold(CASES, wrong_at=(1, 2, 7))
    replies[3] = "我不知道该用哪个工具"
    replies[8] = ToolCall("b", CASES[8].gold_calls[0].name, {}, raw_arguments="{oops")
    gateway = _gateway(tmp_path, {"good": replies})

    out = run_toolcall(ws, cfg, gateway, "good", CASES, seed=0)
    assert (out.n, out.valid, out.matched, out.skipped) == (12, 10, 7, False)

    rows = ws.db.execute("SELECT * FROM ai_cases WHERE eval_id = ? ORDER BY rowid", (out.eval_id,)).fetchall()
    assert [r["case_id"] for r in rows] == [c.id for c in CASES]
    assert [r["passed"] for r in rows] == [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    assert json.loads(rows[3]["scores_json"]) == {
        "cached": False,
        "match": False,
        "reason": "没有工具调用",
        "valid": False,
    }
    assert json.loads(rows[8]["scores_json"])["reason"] == "调用不合法：参数不是合法的 JSON 对象"
    assert (rows[0]["family"], rows[0]["bucket"], rows[0]["steps"]) == (CASES[0].family, "seen", 1)

    record = ws.db.execute("SELECT * FROM ai_evals WHERE id = ?", (out.eval_id,)).fetchone()
    assert (record["suite"], record["model"], record["seed"], record["status"]) == (
        "toolcall",
        "good",
        0,
        "done",
    )
    assert record["suite_fp"] == suite_fingerprint(CASES)
    assert record["setup"] == setup_id("zeroshot", SYSTEM_PROMPT, tool_specs())

    run_dir = ws.root / "runs" / record["run_id"]
    lines = [json.loads(x) for x in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 12 and lines[3]["completion"]["text"] == "我不知道该用哪个工具"
    assert lines[0]["messages"][0] == {
        "role": "system",
        "content": SYSTEM_PROMPT,
        "tool_calls": [],
        "tool_call_id": None,
    }
    assert lines[0]["messages"][1]["content"] == CASES[0].turns[0]
    assert len(read_calls(run_dir / "llm_calls.jsonl")) == 12
    assert gateway.calls_path is None  # 用完还原
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert (meta["kind"], meta["status"], meta["model"], meta["n_cases"]) == ("aieval", "done", "good", 12)


def test_requests_carry_the_shared_prompt_tools_and_seed(ws, cfg, tmp_path):
    gateway = _gateway(tmp_path, {"good": _gold(CASES)})
    run_toolcall(ws, cfg, gateway, "good", CASES, seed=5)
    model = gateway._models["good"]
    assert isinstance(model, ScriptedModel)
    request = model.requests[0]
    assert (request.temperature, request.seed, request.max_tokens) == (0.0, 5, 512)
    assert request.tools == tuple(tool_specs()) and request.messages[0].content == SYSTEM_PROMPT


def test_the_same_eval_is_not_recorded_twice(ws, cfg, tmp_path):
    gateway = _gateway(tmp_path, {"good": _gold(CASES, wrong_at=(1,)) + _gold(CASES)})
    first = run_toolcall(ws, cfg, gateway, "good", CASES, seed=0)
    again = run_toolcall(ws, cfg, gateway, "good", CASES, seed=0)
    assert (again.eval_id, again.valid, again.matched, again.skipped) == (first.eval_id, 12, 11, True)
    other_seed = run_toolcall(ws, cfg, gateway, "good", CASES, seed=1)  # 另一个种子是另一次评测
    assert other_seed.eval_id != first.eval_id and other_seed.matched == 12
    assert ws.db.execute("SELECT COUNT(*) FROM ai_evals").fetchone()[0] == 2

    forced = run_toolcall(ws, cfg, gateway, "good", CASES, seed=0, force=True)  # 重跑：请求相同，命中缓存
    assert forced.matched == 11 and not forced.skipped
    assert ws.db.execute("SELECT COUNT(*) FROM ai_evals").fetchone()[0] == 2
    cached = [
        json.loads(r[0])["cached"]
        for r in ws.db.execute("SELECT scores_json FROM ai_cases WHERE eval_id = ?", (forced.eval_id,))
    ]
    assert cached == [True] * 12


def test_a_failing_model_aborts_the_eval_instead_of_counting_as_wrong(ws, cfg, tmp_path):
    gateway = _gateway(tmp_path, {"good": _gold(CASES)[:5]}, retries=0)  # 脚本只够 5 条
    with pytest.raises(ModelError):
        run_toolcall(ws, cfg, gateway, "good", CASES)
    record = ws.db.execute("SELECT status, run_id FROM ai_evals").fetchone()
    assert record["status"] == "running"
    assert (
        ws.db.execute("SELECT status FROM runs WHERE id = ?", (record["run_id"],)).fetchone()[0] == "failed"
    )
    assert summarize(ws, "toolcall") == []  # 没跑完的不进汇总
    assert ws.db.execute("SELECT COUNT(*) FROM ai_cases").fetchone()[0] == 0

    gateway = _gateway(tmp_path / "second", {"good": _gold(CASES)})
    assert run_toolcall(ws, cfg, gateway, "good", CASES).matched == 12  # 重来时把半截的记录换掉
    assert ws.db.execute("SELECT COUNT(*) FROM ai_evals").fetchone()[0] == 1


def test_evals_do_not_fall_back_to_another_model(ws, cfg, tmp_path):
    gateway = _gateway(tmp_path, {"good": [], "weak": []}, fallback="weak")
    with pytest.raises(ValueError, match="不能配置备用模型"):
        run_toolcall(ws, cfg, gateway, "good", CASES)


# ---- 汇总与配对比较：对手算例子 ----


def _timed(call: ToolCall, ms: float) -> Completion:
    return Completion("", (call,), tokens_in=100, tokens_out=10, latency_ms=ms, cached=False)


def test_summary_and_comparison_hand_computed(ws, cfg, tmp_path):
    # good：种子 0 错位置 1；种子 1 错位置 1、2；种子 2 全对。weak：三个种子都错位置 1、6、7、8
    good = _gold(CASES, wrong_at=(1,)) + _gold(CASES, wrong_at=(1, 2)) + _gold(CASES)
    weak = _gold(CASES, wrong_at=(1, 6, 7, 8)) * 3
    good = [_timed(call, 10.0 * (i + 1)) for i, call in enumerate(good)]  # 延迟 10、20、…、360 毫秒
    gateway = _gateway(tmp_path, {"good": good, "weak": weak})
    for model in ("good", "weak"):
        for seed in (0, 1, 2):
            run_toolcall(ws, cfg, gateway, model, CASES, seed=seed)

    table = {(s.model, s.bucket): s for s in summarize(ws, "toolcall")}
    assert set(table) == {(m, b) for m in ("good", "weak") for b in ("seen", "heldout", ALL)}
    everything = table["good", ALL]
    assert (everything.match.k, everything.match.n, everything.n_cases, everything.seeds) == (
        33,
        36,
        12,
        (0, 1, 2),
    )
    assert everything.match.seed_spread == pytest.approx(12 / 12 - 10 / 12)
    assert (everything.valid.k, everything.valid.seed_spread) == (36, 0.0)
    # 区间按 12 个用例算，不按 36 行算：比例 33/36 = 11/12，11/12 的 Wilson 区间是 [0.6461, 0.9851]
    # （按 36 行算会得到 [0.7817, 0.9713]，窄了将近一半）
    assert (everything.match.n_cases, everything.valid.n_cases) == (12, 12)
    assert everything.match.low == pytest.approx(0.6461, abs=1e-4)
    assert everything.match.high == pytest.approx(0.9851, abs=1e-4)
    # weak 的三个种子逐条相同（24/36）：区间必须和只跑一个种子（8/12）时一样，重复三遍不该让它变窄
    weak_all = table["weak", ALL].match
    assert (weak_all.k, weak_all.n, weak_all.seed_spread) == (24, 36, 0.0)
    assert (weak_all.low, weak_all.high) == pytest.approx(wilson_interval(8, 12))
    seen = table["good", "seen"]
    assert (seen.match.k, seen.match.n) == (15, 18) and table["good", "heldout"].match.k == 18
    # 36 次调用的延迟是 10…360：中位数是第 18 个（180），95 分位是第 ceil(34.2) = 35 个（350）
    assert (everything.n_timed, everything.latency_p50_ms, everything.latency_p95_ms) == (36, 180.0, 350.0)
    assert (everything.tokens_in_mean, everything.tokens_out_mean) == (100.0, 10.0)
    assert (table["weak", "heldout"].match.k, table["weak", "seen"].match.k, table["weak", ALL].match.k) == (
        9,
        15,
        24,
    )

    pairs = {(c.bucket, c.metric): c for c in compare(ws, "toolcall")}
    overall = pairs[ALL, "match"]
    # 按多数种子：good 只错位置 1（三个种子里错两次）；位置 2 只错一次，算通过。weak 错 1、6、7、8
    assert (overall.model_a, overall.model_b, overall.n_cases) == ("good", "weak", 12)
    assert (overall.pass_a, overall.pass_b, overall.only_a, overall.only_b) == (11, 8, 3, 0)
    assert overall.p_value == pytest.approx(0.25)  # 2 · (1/2)³
    assert (pairs["seen", "match"].only_a, pairs["seen", "match"].only_b, pairs["seen", "match"].p_value) == (
        0,
        0,
        1.0,
    )
    assert pairs["heldout", "match"].only_a == 3 and pairs[ALL, "valid"].p_value == 1.0

    report = render_report(ws, "toolcall")
    assert (
        "| good | " in report and "33/36 = 0.917 [0.646, 0.985]" in report and "180 / 350（36 次）" in report
    )
    assert "| 11 | 8 | 3 | 0 | 0.25 |" in report
    assert f"用例集 {suite_fingerprint(CASES)[:12]}（12 条用例）" in report


def test_cached_calls_are_left_out_of_latency(ws, cfg, tmp_path):
    replies = [_timed(c, 100.0) for c in _gold(CASES)]
    gateway = _gateway(tmp_path, {"good": replies})
    run_toolcall(ws, cfg, gateway, "good", CASES, seed=0)
    run_toolcall(ws, cfg, gateway, "good", CASES, seed=0, force=True)  # 全部命中缓存
    (summary,) = [s for s in summarize(ws, "toolcall") if s.bucket == ALL]
    assert (summary.n_timed, summary.latency_p50_ms, summary.match.k) == (0, None, 12)
    assert "— / —（0 次）" in render_report(ws, "toolcall")


def test_results_on_different_case_sets_are_not_compared(ws, cfg, tmp_path):
    gateway = _gateway(tmp_path, {"good": _gold(CASES), "weak": _gold(CASES[:6])})
    run_toolcall(ws, cfg, gateway, "good", CASES)
    run_toolcall(ws, cfg, gateway, "weak", CASES[:6])
    assert compare(ws, "toolcall") == []
    assert render_report(ws, "toolcall").count("## 用例集") == 2


def test_report_says_so_when_there_is_nothing(ws):
    assert render_report(ws, "toolcall") == "套件 toolcall 还没有评测结果。先运行 roboarena aieval run。\n"
