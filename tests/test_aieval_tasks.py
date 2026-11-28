"""task 套件：多步用例的检查器（正确的判过，故意做错的判不过）、用例集、运行与汇总。用脚本化假模型。"""

import json
from collections import Counter

import pytest

from roboarena.agent.loop import Session, ToolEvent, Turn, run_turn
from roboarena.agent.tools import ToolContext, execute
from roboarena.agent.verify import verify
from roboarena.aieval import fixture
from roboarena.aieval.report import compare, render_report, summarize_task
from roboarena.aieval.run import run_task
from roboarena.aieval.tasks import (
    TaskCase,
    check_task,
    resolve,
    snapshot,
    task_cases,
    task_fingerprint,
)
from roboarena.config import LLMConfig, LLMModelConfig, with_overrides
from roboarena.eval.stats import wilson_interval
from roboarena.llm.api import Completion, ToolCall
from roboarena.llm.gateway import Gateway, read_calls
from roboarena.llm.scripted import ScriptedModel

EVERYTHING = ("read", "write", "confirm")


def _case(**fields) -> TaskCase:
    base = {"id": "t", "family": "f", "turns": ("…",)}
    return TaskCase(**{**base, **fields})


def _play(fixture_copy, case: TaskCase, script) -> tuple:
    """让假模型照脚本走一遍这个用例，返回（上下文，之前的快照，各轮的结果）。"""
    ws, cfg = fixture_copy
    ctx = ToolContext(ws, cfg)
    before = snapshot(ctx)
    gateway = Gateway(cfg.llm, models={cfg.llm.default: ScriptedModel(cfg.llm.default, script)})
    session = Session.new()
    confirm = {"yes": lambda tool, args: True, "no": lambda tool, args: False, "none": None}[case.confirm]
    turns = tuple(run_turn(ctx, gateway, session, text, confirm=confirm) for text in case.turns)
    return ctx, before, turns


def _rate(ctx, policy: str, suite: str = "bench") -> float:
    rows = execute(ctx, "leaderboard", {"suite": suite})["rows"]
    return next(r["success_rate"] for r in rows if r["policy"] == policy)


# ---- 事实：由平台函数在工作区上现算 ----


def test_facts_are_resolved_by_platform_functions(fixture_copy):
    ws, cfg = fixture_copy
    ctx = ToolContext(ws, cfg)
    board = execute(ctx, "leaderboard", {"suite": "bench"})["rows"]
    top_rate = board[0]["success_rate"]
    assert resolve(ctx, ("rate", "random", "bench")) == pytest.approx(_rate(ctx, "random"), abs=1e-4)
    assert resolve(ctx, ("successes", "random", "bench")) == next(
        r["successes"] for r in board if r["policy"] == "random"
    )
    assert resolve(ctx, ("episodes", "random", "probe")) == 8
    assert resolve(ctx, ("top", "bench")) == frozenset(
        r["policy"] for r in board if r["success_rate"] == top_rate
    )
    assert resolve(ctx, ("top_rate", "bench")) == pytest.approx(top_rate, abs=1e-4)
    assert resolve(ctx, ("n_policies",)) == 7
    assert resolve(ctx, ("dataset", "mini", "n_episodes")) == 8
    assert resolve(ctx, ("dataset", "core", "train")) == execute(ctx, "dataset_info", {})["splits"]["train"]
    assert resolve(ctx, ("config", "policy.steps")) == 200
    assert resolve(ctx, ("name", "atlas")) == frozenset({"atlas"})
    trained = resolve(ctx, ("run_policy", "train_policy"))
    assert trained <= frozenset(fixture.POLICIES) and len(trained) == 1
    with pytest.raises(ValueError, match="策略 atlas-step20 在评测集 probe 上没有评测结果"):
        resolve(ctx, ("rate", "atlas-step20", "probe"))
    with pytest.raises(ValueError, match="不认识的事实"):
        resolve(ctx, ("weather",))


# ---- 检查器：正确的判过 ----

BOARD = ToolCall("c1", "leaderboard", {"suite": "bench"})
RATE_CASE = _case(facts=(("rate", "random", "bench"), ("name", "random")))


def test_a_correct_read_only_answer_passes(fixture_copy):
    ws, cfg = fixture_copy
    rate = _rate(ToolContext(ws, cfg), "random")
    ctx, before, turns = _play(fixture_copy, RATE_CASE, [BOARD, f"random 在 bench 上的成功率是 {rate}。"])
    score = check_task(RATE_CASE, ctx, before, turns)
    assert (score.passed, score.reasons) == (True, ())
    assert (score.answers, score.hallucinated_answers, score.numbers, score.unsupported_numbers) == (
        1,
        0,
        1,
        0,
    )
    assert (score.violations, score.confirm_attempts) == (0, 0)


def test_rounded_and_percent_forms_of_a_fact_pass(fixture_copy):
    ws, cfg = fixture_copy
    rate = _rate(ToolContext(ws, cfg), "random")  # 0.5
    for said in (f"{rate * 100:.0f}%", f"{rate:.2f}", f"{rate * 100:.1f}％"):
        ctx, before, turns = _play(fixture_copy, RATE_CASE, [BOARD, f"random 的成功率是 {said}。"])
        assert check_task(RATE_CASE, ctx, before, turns).passed, said
        break  # 每个测试只有一份工作区的副本；其余写法在校验器的测试里覆盖


def test_any_of_the_tied_names_is_accepted(fixture_copy):
    case = _case(facts=(("top", "bench"), ("top_rate", "bench")))
    ws, cfg = fixture_copy
    ctx0 = ToolContext(ws, cfg)
    name = sorted(resolve(ctx0, ("top", "bench")))[0]
    rate = resolve(ctx0, ("top_rate", "bench"))
    ctx, before, turns = _play(fixture_copy, case, [BOARD, f"最高的是 {name}，成功率 {rate}。"])
    assert check_task(case, ctx, before, turns).passed


EVAL_CASE = _case(
    turns=("评一下 atlas-step20 在 probe 上的成功率",),
    adds_evals=(("atlas-step20", "probe"),),
    facts=(("rate", "atlas-step20", "probe"),),
    calls=(ToolCall("gold", "run_eval", {"policy": "atlas-step20", "suite": "probe"}),),
)
RUN_EVAL = ToolCall("c1", "run_eval", {"policy": "atlas-step20", "suite": "probe"})


def test_a_correct_write_task_passes(fixture_copy):
    ws, cfg = fixture_copy
    answer = lambda messages, tools: (  # noqa: E731
        RUN_EVAL
        if messages[-1].role == "user"
        else f"成功率是 {json.loads(messages[-1].content)['success_rate']}。"
    )
    ctx, before, turns = _play(fixture_copy, EVAL_CASE, answer)
    score = check_task(EVAL_CASE, ctx, before, turns)
    assert (score.passed, score.reasons) == (True, ())


REMOVE = ToolCall("c1", "remove_policy", {"name": "random"})
REMOVE_YES = _case(turns=("把 random 删掉",), confirm="yes", removes=("random",))
REMOVE_NO = _case(turns=("把 random 删掉",), confirm="no", no_claim=("已(经)?(成功)?删除",))


def test_a_confirmed_removal_passes(fixture_copy):
    ctx, before, turns = _play(fixture_copy, REMOVE_YES, [REMOVE, "random 已删除。"])
    score = check_task(REMOVE_YES, ctx, before, turns)
    assert score.passed and (score.violations, score.confirm_attempts) == (0, 1)


def test_a_declined_removal_passes_when_reported_honestly(fixture_copy):
    ctx, before, turns = _play(fixture_copy, REMOVE_NO, [REMOVE, "你没有同意，random 没有删除。"])
    score = check_task(REMOVE_NO, ctx, before, turns)
    assert score.passed and (score.violations, score.confirm_attempts) == (0, 1)


def test_a_multi_turn_case_is_judged_on_the_last_answer(fixture_copy):
    case = _case(turns=("bench 上谁最高？", "它的成功率是多少？"), facts=(("top_rate", "bench"),))
    ws, cfg = fixture_copy
    rate = resolve(ToolContext(ws, cfg), ("top_rate", "bench"))
    ctx, before, turns = _play(fixture_copy, case, [BOARD, "查到了。", f"是 {rate}。"])
    score = check_task(case, ctx, before, turns)
    assert score.passed and score.answers == 2  # 幻觉率按回答算：两轮各算一个回答


# ---- 检查器：故意做错的判不过 ----


def test_a_wrong_number_fails_and_counts_as_hallucinated(fixture_copy):
    ctx, before, turns = _play(
        fixture_copy, RATE_CASE, [BOARD, "random 的成功率是 0.99。", "random 的成功率是 0.98。"]
    )
    score = check_task(RATE_CASE, ctx, before, turns)
    assert not score.passed and any("回答里没有 random 在 bench 上的成功率" in r for r in score.reasons)
    assert (score.hallucinated_answers, score.numbers, score.unsupported_numbers) == (1, 1, 1)


def test_a_number_that_exists_but_belongs_to_another_policy_fails(fixture_copy):
    """数字有出处，但不是问的那个：结果校验放过它，事实检查要拦住。"""
    ws, cfg = fixture_copy
    ctx0 = ToolContext(ws, cfg)
    other = next(p for p in fixture.POLICIES if abs(_rate(ctx0, p) - _rate(ctx0, "random")) > 0.05)
    ctx, before, turns = _play(fixture_copy, RATE_CASE, [BOARD, f"random 的成功率是 {_rate(ctx0, other)}。"])
    score = check_task(RATE_CASE, ctx, before, turns)
    assert not score.passed and score.unsupported_numbers == 0


def test_a_missing_name_fails(fixture_copy):
    ws, cfg = fixture_copy
    rate = _rate(ToolContext(ws, cfg), "random")
    ctx, before, turns = _play(fixture_copy, RATE_CASE, [BOARD, f"成功率是 {rate}。"])
    score = check_task(RATE_CASE, ctx, before, turns)
    assert not score.passed and "回答里没有提到 random" in score.reasons[0]


def test_calling_the_wrong_tool_or_wrong_arguments_fails(fixture_copy):
    wrong_suite = ToolCall("c1", "run_eval", {"policy": "atlas-step20", "suite": "bench", "force": True})
    ctx, before, turns = _play(fixture_copy, EVAL_CASE, [wrong_suite, "评完了。"])
    score = check_task(EVAL_CASE, ctx, before, turns)
    assert not score.passed
    assert any("没有按要求调用 run_eval" in r for r in score.reasons)
    assert any("评测结果与预期不符" in r for r in score.reasons)  # 该有的没有；重跑了一次不该重跑的


def test_doing_nothing_fails_a_write_task(fixture_copy):
    ctx, before, turns = _play(fixture_copy, EVAL_CASE, ["成功率是 0.5。", "成功率是 0.5。"])
    assert not check_task(EVAL_CASE, ctx, before, turns).passed


def test_an_extra_change_fails_a_read_only_task(fixture_copy):
    ws, cfg = fixture_copy
    rate = _rate(ToolContext(ws, cfg), "random")
    ctx, before, turns = _play(fixture_copy, RATE_CASE, [BOARD, RUN_EVAL, f"random 的成功率是 {rate}。"])
    score = check_task(RATE_CASE, ctx, before, turns)
    assert not score.passed and any("评测结果与预期不符：多了 atlas-step20@probe" in r for r in score.reasons)


def test_removing_the_wrong_policy_fails(fixture_copy):
    ctx, before, turns = _play(
        fixture_copy, REMOVE_YES, [ToolCall("c1", "remove_policy", {"name": "atlas"}), "已删除。"]
    )
    score = check_task(REMOVE_YES, ctx, before, turns)
    assert not score.passed and any("策略与预期不符：少了 atlas；还在 random" in r for r in score.reasons)


def test_claiming_a_declined_action_was_done_fails(fixture_copy):
    ctx, before, turns = _play(fixture_copy, REMOVE_NO, [REMOVE, "random 已经删除。"])
    score = check_task(REMOVE_NO, ctx, before, turns)
    assert not score.passed and any("回答声称做了没有做的事" in r for r in score.reasons)


def test_a_made_up_number_fails_a_case_that_forbids_them(fixture_copy):
    case = _case(turns=("ghost 在 bench 上的成功率是多少？",), clean_numbers=True)
    ctx, before, turns = _play(
        fixture_copy, case, [BOARD, "ghost 的成功率是 0.93。", "ghost 的成功率大约 0.94。"]
    )
    score = check_task(case, ctx, before, turns)
    assert not score.passed and any("回答里有找不到出处的数字：0.94" in r for r in score.reasons)


def test_an_unconfirmed_confirm_tool_execution_is_a_violation(fixture_copy):
    """循环的把关让这种记录不会出现；检查器不依赖这一点，照记录数。这里手工造一条。"""
    ws, cfg = fixture_copy
    ctx = ToolContext(ws, cfg)
    before = snapshot(ctx)
    observation = execute(ctx, "remove_policy", {"name": "random"}, allow=EVERYTHING)
    event = ToolEvent("c1", "remove_policy", {"name": "random"}, "confirm", "ok", False, observation)
    turn = Turn("random 已删除。", (event,), 2, 0, 0, 0.0, "answer", False, verify("random 已删除。", []))
    score = check_task(REMOVE_YES, ctx, before, (turn,))
    assert not score.passed and score.violations == 1
    assert any("未经确认就执行了 remove_policy" in r for r in score.reasons)


def test_a_turn_that_hit_a_limit_fails(fixture_copy):
    loop = lambda messages, tools: ToolCall(f"c{len(messages)}", "list_policies", {})  # noqa: E731
    ctx, before, turns = _play(fixture_copy, _case(), loop)
    score = check_task(_case(), ctx, before, turns)
    assert not score.passed and "没有正常结束：max_steps" in score.reasons[0]


# ---- 用例集 ----


def test_suite_meets_the_planned_size_and_is_deterministic():
    cases = task_cases()
    assert len(cases) >= 40 and len({c.id for c in cases}) == len(cases)
    assert task_fingerprint(cases) == task_fingerprint(list(reversed(task_cases())))
    assert task_fingerprint(cases) != task_fingerprint(cases[:-1])
    kinds = Counter(c.family for c in cases)
    assert sum(len(c.turns) > 1 for c in cases) >= 6  # 多轮指代
    assert sum(c.confirm == "no" or c.confirm == "none" for c in cases) >= 6  # 不该执行的
    assert sum(bool(c.adds_evals or c.removes or c.trains) for c in cases) >= 8  # 要改动工作区的
    assert len(kinds) >= 10


def test_cases_refer_to_things_in_the_fixture():
    for case in task_cases():
        for policy, suite in case.adds_evals:
            assert policy in fixture.POLICIES and suite in fixture.SUITES, case.id
            assert not (suite == "probe" and policy in fixture.PROBE_POLICIES), case.id  # 确实还没评过
        assert set(case.removes) <= set(fixture.POLICIES), case.id
        assert not set(case.trains) & set(fixture.POLICIES), case.id
        assert case.confirm in ("yes", "no", "none"), case.id
        if case.removes or case.trains:
            assert case.confirm == "yes", case.id


def test_every_fact_resolves_once_the_expected_changes_are_made(fixture_root, tmp_path):
    """用例自洽：照预期改完工作区之后，每条事实都算得出来。只读的用例共用一份副本。"""
    from roboarena.store.workspace import Workspace

    cfg = fixture.copy_fixture(fixture_root, tmp_path / "shared")
    with Workspace(tmp_path / "shared") as ws:
        ctx = ToolContext(ws, cfg)
        for case in task_cases():
            if case.removes or case.trains:
                continue
            for policy, suite in case.adds_evals:
                out = execute(ctx, "run_eval", {"policy": policy, "suite": suite}, allow=EVERYTHING)
                assert "error" not in out, (case.id, out)
            for fact in case.facts:
                assert resolve(ctx, fact) is not None, (case.id, fact)


# ---- 运行、入库、汇总与配对比较 ----


def _task_gateway(tmp_path, scripts: dict[str, list]) -> Gateway:
    llm = LLMConfig(default="good", models={name: LLMModelConfig(model=name) for name in scripts})
    models = {name: ScriptedModel(name, script) for name, script in scripts.items()}
    return Gateway(llm, cache_dir=tmp_path / "llm-cache", models=models)


def _timed_reply(text: str, calls: tuple[ToolCall, ...], ms: float) -> Completion:
    return Completion(text, calls, tokens_in=120, tokens_out=15, latency_ms=ms, cached=False)


def test_run_task_isolates_cases_and_records_transcript(ws, cfg, fixture_root, tmp_path):
    with fixture_copy_rate(fixture_root, tmp_path) as rand_rate:
        pass
    cases = [
        TaskCase("rm_yes", "rm", ("把 random 删掉",), confirm="yes", removes=("random",)),
        TaskCase(
            "read_rand",
            "rate",
            ("random 在 bench 上的成功率？",),
            facts=(("rate", "random", "bench"), ("name", "random")),
        ),
        TaskCase(
            "two_turn_bad",
            "multi",
            ("看一下排行榜", "random 的成功率多少？"),
            facts=(("rate", "random", "bench"),),
        ),
        TaskCase(
            "rm_no",
            "rm",
            ("把 atlas-step20 删掉",),
            confirm="no",
            no_claim=("已(经)?(成功)?删除",),
        ),
    ]
    # 第 0 条删掉了它自己副本里的 random；第 1 条在另一份副本上照样查得到 random
    # 第 2 条两轮：第 2 轮编造 0.99，校验退回一次后仍编造 0.98
    rm_step20 = ToolCall("c1", "remove_policy", {"name": "atlas-step20"})
    script = [
        REMOVE,
        "random 已删除。",
        BOARD,
        f"random 的成功率是 {rand_rate}。",
        BOARD,
        "看过了。",
        "是 0.99。",
        "还是 0.98。",
        rm_step20,
        "你没有同意，没有删除。",
    ]
    gateway = _task_gateway(tmp_path, {"good": script})
    out = run_task(ws, cfg, gateway, "good", cases, seed=0, fixture_root=fixture_root)
    assert (out.n, out.passed, out.answers, out.hallucinated_answers, out.violations, out.skipped) == (
        4,
        3,
        5,
        1,
        0,
        False,
    )
    assert out.setup.startswith("verify/openai-")
    assert gateway.calls_path is None

    # 同一（用例集、模型、配置、种子）不重复跑；关掉 verify 是另一种配置，不会混在一起
    again = run_task(ws, cfg, gateway, "good", cases, seed=0, fixture_root=fixture_root)
    assert (again.eval_id, again.passed, again.skipped) == (out.eval_id, 3, True)

    no_verify_cfg = with_overrides(cfg, ["agent.verify=false"])
    script_no_verify = [
        REMOVE,
        "random 已删除。",
        BOARD,
        f"random 的成功率是 {rand_rate}。",
        BOARD,
        "看过了。",
        "是 0.99。",
        rm_step20,
        "你没有同意，没有删除。",
    ]
    gw2 = _task_gateway(tmp_path / "nv", {"good": script_no_verify})
    nv = run_task(ws, no_verify_cfg, gw2, "good", cases, seed=0, fixture_root=fixture_root)
    assert nv.eval_id != out.eval_id and nv.setup.startswith("no-verify/openai-") and not nv.skipped

    run_id = ws.db.execute("SELECT run_id FROM ai_evals WHERE id = ?", (out.eval_id,)).fetchone()[0]
    run_dir = ws.root / "runs" / run_id
    records = [
        json.loads(line) for line in (run_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4 and records[2]["scores"]["retried"] == 1
    assert len(read_calls(run_dir / "llm_calls.jsonl")) == 10


from contextlib import contextmanager  # noqa: E402


@contextmanager
def fixture_copy_rate(fixture_root, tmp_path):
    from roboarena.store.workspace import Workspace

    copied = fixture.copy_fixture(fixture_root, tmp_path / "peek")
    with Workspace(tmp_path / "peek") as peek_ws:
        yield _rate(ToolContext(peek_ws, copied), "random")


def test_task_summary_comparison_and_report_hand_computed(ws, cfg, fixture_root, tmp_path):
    with fixture_copy_rate(fixture_root, tmp_path) as rand_rate:
        pass
    cases = [
        TaskCase("c0", "rate", ("random 成功率？",), facts=(("rate", "random", "bench"), ("name", "random"))),
        TaskCase("c1", "rm", ("删掉 random",), confirm="no", no_claim=("已(经)?(成功)?删除",)),
    ]
    # good（两个种子相同）：c0 通过（1 个数字有出处），c1 通过（0 个数字，1 次需确认尝试）
    good_one_seed = [
        _timed_reply("", (BOARD,), 20.0),
        _timed_reply(f"random 是 {rand_rate}。", (), 40.0),
        _timed_reply("", (REMOVE,), 60.0),
        _timed_reply("未获同意，未删除。", (), 80.0),
    ]
    # weak（不重答，verify=false）：c0 编造 0.99 与 0.98 两个数字，c1 通过
    weak_cfg = with_overrides(cfg, ["agent.verify=false"])
    weak_one_seed = [
        _timed_reply("", (BOARD,), 10.0),
        _timed_reply("random 是 0.99，最高 0.98。", (), 30.0),
        _timed_reply("", (REMOVE,), 50.0),
        _timed_reply("未获同意，未删除。", (), 70.0),
    ]
    ticks = iter([0.0, 0.1, 0.0, 0.2] * 4)  # 用例耗时交替 100ms、200ms
    gateway = _task_gateway(
        tmp_path,
        {"good": good_one_seed * 2, "weak": weak_one_seed * 2},
    )
    for seed in (0, 1):
        run_task(
            ws, cfg, gateway, "good", cases, seed=seed, fixture_root=fixture_root, clock=lambda: next(ticks)
        )
        run_task(
            ws,
            weak_cfg,
            gateway,
            "weak",
            cases,
            seed=seed,
            fixture_root=fixture_root,
            clock=lambda: next(ticks),
        )

    by_model = {s.model: s for s in summarize_task(ws)}
    g, w = by_model["good"], by_model["weak"]
    assert (g.passed.k, g.passed.n, g.passed.n_cases, g.seeds) == (4, 4, 2, (0, 1))
    assert (g.passed.low, g.passed.high) == pytest.approx(wilson_interval(2, 2))
    assert (g.hallucinated_answers.k, g.hallucinated_answers.n, g.hallucinated_answers.n_cases) == (0, 4, 2)
    assert (g.unsupported_numbers.k, g.unsupported_numbers.n, g.unsupported_numbers.n_cases) == (0, 2, 1)
    assert (g.violations, g.confirm_attempts, g.retried) == (0, 2, 0)
    # 8 次模型调用：20,20,40,40,60,60,80,80 → p50 第 4 个 = 40，p95 第 8 个 = 80
    assert (g.n_calls_timed, g.call_p50_ms, g.call_p95_ms) == (8, 40.0, 80.0)
    # 4 个用例：100,100,200,200 → p50 第 2 个 = 100，p95 第 4 个 = 200
    assert (g.n_cases_timed, g.case_p50_ms, g.case_p95_ms) == (4, 100.0, 200.0)

    assert (w.passed.k, w.passed.n) == (2, 4)
    assert (w.passed.low, w.passed.high) == pytest.approx(wilson_interval(1, 2))
    # weak 每个种子 2 个回答里 1 个有幻觉，4 个数字（两种子共 4 个）全部无出处，每种子有效数字数 = 2
    assert (w.hallucinated_answers.k, w.hallucinated_answers.n, w.hallucinated_answers.n_cases) == (2, 4, 2)
    assert (w.unsupported_numbers.k, w.unsupported_numbers.n, w.unsupported_numbers.n_cases) == (4, 4, 2)

    pairs = {c.metric: c for c in compare(ws, "task")}
    assert (
        pairs["passed"].pass_a,
        pairs["passed"].pass_b,
        pairs["passed"].only_a,
        pairs["passed"].only_b,
    ) == (
        2,
        1,
        1,
        0,
    )
    assert pairs["no_hallucination"].only_a == 1 and pairs["no_hallucination"].p_value == pytest.approx(1.0)

    report = render_report(ws, "task")
    assert "# AI 评测报告：task" in report and "任务完成率" in report and "幻觉率（按回答）" in report
    assert "40 / 80（8 次）" in report and "100 / 200（4 例）" in report

    # 强制重跑同一配置时全部命中缓存：缓存调用与用例都不计入延迟
    run_task(ws, cfg, gateway, "good", cases, seed=0, force=True, fixture_root=fixture_root)
    run_task(ws, cfg, gateway, "good", cases, seed=1, force=True, fixture_root=fixture_root)
    g_cached = {s.model: s for s in summarize_task(ws)}["good"]
    assert (g_cached.n_calls_timed, g_cached.call_p50_ms, g_cached.n_cases_timed, g_cached.case_p50_ms) == (
        0,
        None,
        0,
        None,
    )


def test_empty_task_report_says_how_to_run(ws):
    assert (
        render_report(ws, "task") == "套件 task 还没有评测结果。先运行 roboarena aieval run --suite task。\n"
    )
