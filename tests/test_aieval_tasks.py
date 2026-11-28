"""task 套件：多步用例的检查器（正确的判过，故意做错的判不过）、用例集、运行与汇总。用脚本化假模型。"""

import json
from collections import Counter

import pytest

from roboarena.agent.loop import Session, ToolEvent, Turn, run_turn
from roboarena.agent.tools import ToolContext, execute
from roboarena.agent.verify import verify
from roboarena.aieval import fixture
from roboarena.aieval.tasks import (
    TaskCase,
    check_task,
    resolve,
    snapshot,
    task_cases,
    task_fingerprint,
)
from roboarena.llm.api import ToolCall
from roboarena.llm.gateway import Gateway
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
