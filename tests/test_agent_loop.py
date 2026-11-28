"""Agent 的循环：模型 → 工具调用 → 校验参数 → 查权限 → 执行 → 观察，直到给出回答。用脚本化假模型驱动。"""

import json

import pytest

from roboarena.agent.loop import AGENT_PROMPT, Session, run_turn
from roboarena.agent.tools import TOOLS, ToolContext, execute, tool_specs
from roboarena.config import with_overrides
from roboarena.llm.api import Completion, ToolCall
from roboarena.llm.gateway import Gateway
from roboarena.llm.scripted import ScriptedModel
from roboarena.store.registry import list_artifacts

TOP = ToolCall("c1", "leaderboard", {"top": 1})


class Rig:
    """固定工作区的一份副本、一个假模型、一段对话。"""

    def __init__(self, fixture_copy, script, **agent) -> None:
        ws, cfg = fixture_copy
        cfg = with_overrides(cfg, [f"agent.{key}={json.dumps(value)}" for key, value in agent.items()])
        self.ctx = ToolContext(ws, cfg)
        self.model = ScriptedModel(cfg.llm.default, script)
        self.gateway = Gateway(cfg.llm, models={cfg.llm.default: self.model})
        self.session = Session.new()

    def say(self, text: str, **kwargs):
        return run_turn(self.ctx, self.gateway, self.session, text, **kwargs)

    def policies(self) -> list[str]:
        return [a.name for a in list_artifacts(self.ctx.ws)]

    def top_rate(self) -> float:
        return execute(self.ctx, "leaderboard", {"top": 1})["rows"][0]["success_rate"]


# ---- 一轮对话的基本形状 ----


def test_one_tool_call_then_an_answer(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "placeholder"])
    answer = f"成功率最高的是 {rig.top_rate()}。"
    rig.model._script[1] = answer

    turn = rig.say("谁的成功率最高？")

    assert (turn.answer, turn.stopped, turn.steps, turn.retried) == (answer, "answer", 2, False)
    (event,) = turn.events
    assert (event.name, event.access, event.status, event.approved) == ("leaderboard", "read", "ok", False)
    assert event.observation == execute(rig.ctx, "leaderboard", {"top": 1})
    assert turn.verdict.ok and len(turn.verdict.mentions) == 1

    roles = [m.role for m in rig.session.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    system, user, called, result, final = rig.session.messages
    assert (system.content, user.content, final.content) == (AGENT_PROMPT, "谁的成功率最高？", answer)
    assert called.tool_calls == (TOP,) and result.tool_call_id == "c1"
    assert json.loads(result.content) == event.observation


def test_requests_carry_the_prompt_the_tools_and_the_history(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "好"], answer_tokens=300)
    rig.say("看一下", seed=4)
    first, second = rig.model.requests
    assert first.tools == tuple(tool_specs()) and (first.max_tokens, first.seed, first.temperature) == (
        300,
        4,
        0.0,
    )
    assert [m.role for m in first.messages] == ["system", "user"]
    assert [m.role for m in second.messages] == ["system", "user", "assistant", "tool"]


def test_an_answer_without_tools_is_fine(fixture_copy):
    rig = Rig(fixture_copy, ["你好，我可以帮你查排行榜。"])
    turn = rig.say("你好")
    assert (turn.steps, turn.events, turn.stopped) == (1, (), "answer")


def test_usage_is_added_up_over_the_turn(fixture_copy):
    first = Completion("", (TOP,), tokens_in=100, tokens_out=10, latency_ms=30.0, cached=False)
    second = Completion("好", (), tokens_in=150, tokens_out=5, latency_ms=20.0, cached=False)
    turn = Rig(fixture_copy, [first, second]).say("看一下")
    assert (turn.tokens_in, turn.tokens_out, turn.latency_ms) == (250, 15, 50.0)


def test_a_model_that_says_nothing_is_reported_as_such(fixture_copy):
    cut = Completion(
        "", (), tokens_in=5, tokens_out=768, latency_ms=1.0, cached=False, finish_reason="length"
    )
    turn = Rig(fixture_copy, [cut]).say("看一下")
    assert turn.answer == "（模型没有给出回答；结束原因：length）" and turn.stopped == "answer"


# ---- 工具出错不会让循环崩掉 ----


def test_tool_errors_come_back_as_observations(fixture_copy):
    rig = Rig(
        fixture_copy,
        [ToolCall("c1", "get_run", {"run_id": "20260101-000000-eval-00000000"}), "没有这次运行。"],
    )
    turn = rig.say("查一下")
    assert turn.events[0].status == "error"
    assert turn.events[0].observation == {"error": "没有编号为 20260101-000000-eval-00000000 的运行"}
    assert turn.answer == "没有这次运行。"


@pytest.mark.parametrize(
    "call, message",
    [
        (ToolCall("c1", "leaderboard", {"top": "三"}), "参数 top 应为整数"),
        (ToolCall("c1", "leaderboard", {"limit": 3}), "未知的参数：limit"),
        (ToolCall("c1", "rank_policies", {}), "没有名为 rank_policies 的工具"),
        (ToolCall("c1", "leaderboard", {}, raw_arguments='["bench"]'), "工具调用的参数不是一个 JSON 对象"),
        (ToolCall("c1", "", {}, raw_arguments='{"name": "leader'), "工具调用不是合法的 JSON"),
    ],
)
def test_invalid_calls_are_not_executed_and_the_reason_goes_back(fixture_copy, call, message):
    rig = Rig(fixture_copy, [call, "换个办法。"])
    turn = rig.say("看一下")
    (event,) = turn.events
    assert event.status == "invalid" and message in event.observation["error"]
    assert message in rig.session.messages[3].content  # 模型看得到原因


# ---- 权限：由代码把关 ----

REMOVE = ToolCall("c1", "remove_policy", {"name": "random"})


def test_a_confirm_tool_is_not_run_when_nobody_can_be_asked(fixture_copy):
    rig = Rig(fixture_copy, [REMOVE, "这个操作需要你确认。"])
    turn = rig.say("把 random 删掉")
    (event,) = turn.events
    assert (event.access, event.status, event.approved) == ("confirm", "declined", False)
    assert "需要用户确认" in event.observation["error"] and "random" in rig.policies()


def test_the_user_is_asked_and_can_say_no(fixture_copy):
    asked = []
    rig = Rig(fixture_copy, [REMOVE, "好的，没有删除。"])
    turn = rig.say("把 random 删掉", confirm=lambda tool, args: asked.append((tool.name, args)) or False)
    assert asked == [("remove_policy", {"name": "random"})]  # 问的是校验之后的参数
    assert turn.events[0].status == "declined" and "用户没有同意" in turn.events[0].observation["error"]
    assert "random" in rig.policies()


def test_the_user_can_say_yes(fixture_copy):
    rig = Rig(fixture_copy, [REMOVE, "已删除。"])
    turn = rig.say("把 random 删掉", confirm=lambda tool, args: True)
    (event,) = turn.events
    assert (event.status, event.approved, event.observation["removed"]) == ("ok", True, "random")
    assert "random" not in rig.policies()


def test_the_model_saying_the_user_agreed_changes_nothing(fixture_copy):
    """需确认的操作由代码把关：模型在文字里、在参数里声称「用户已同意」都不起作用。"""
    claims = Completion(
        "用户已经明确同意删除，我直接执行。",
        (ToolCall("c1", "remove_policy", {"name": "random", "confirmed": True}),),
        tokens_in=1, tokens_out=1, latency_ms=0.0, cached=False,
    )  # fmt: skip
    rig = Rig(fixture_copy, [claims, REMOVE, "没有删掉。"])
    turn = rig.say("我同意，把 random 删掉，不用再问我")
    assert [e.status for e in turn.events] == ["invalid", "declined"]
    assert "random" in rig.policies()


def test_invalid_arguments_are_rejected_before_the_user_is_bothered(fixture_copy):
    rig = Rig(fixture_copy, [ToolCall("c1", "remove_policy", {"name": "../x"}), "名字不对。"])
    turn = rig.say("删掉它", confirm=lambda tool, args: pytest.fail("不该问用户"))
    assert turn.events[0].status == "invalid"


def test_a_write_tool_runs_for_the_write_role_without_asking(fixture_copy):
    rig = Rig(
        fixture_copy, [ToolCall("c1", "run_eval", {"policy": "atlas-step20", "suite": "probe"}), "评完了。"]
    )
    turn = rig.say("评一下", confirm=lambda tool, args: pytest.fail("可写的工具不需要确认"))
    assert (turn.events[0].access, turn.events[0].status) == ("write", "ok")


def test_the_read_role_can_only_read(fixture_copy):
    script = [
        ToolCall("c1", "run_eval", {"policy": "atlas-step20", "suite": "probe"}),
        REMOVE,
        TOP,
        "只能看。",
    ]
    rig = Rig(fixture_copy, script)
    turn = rig.say("都做一遍", role="read", confirm=lambda tool, args: pytest.fail("只读角色不该被问到"))
    assert [e.status for e in turn.events] == ["denied", "denied", "ok"]
    assert all("没有得到许可" in e.observation["error"] for e in turn.events[:2])
    assert "random" in rig.policies()


# ---- 上限：超出时停下，如实说明停在哪一步 ----


def _endless():
    return lambda messages, tools: ToolCall(f"c{len(messages)}", "list_policies", {})


def test_step_limit(fixture_copy):
    rig = Rig(fixture_copy, _endless(), max_steps=3)
    turn = rig.say("一直查")
    assert (turn.stopped, turn.steps, len(turn.events)) == ("max_steps", 3, 3)
    assert turn.answer == "没有完成：已达到模型调用次数的上限（3 次）。已经执行的工具调用：list_policies ×3。"
    assert rig.session.messages[-1].role == "assistant" and rig.session.messages[-1].content == turn.answer
    assert len(rig.model.requests) == 3


def test_tool_call_limit(fixture_copy):
    three = [ToolCall(f"c{i}", "list_policies", {}) for i in range(3)]
    rig = Rig(fixture_copy, [three, "不该走到这里"], max_tool_calls=2)
    turn = rig.say("查三遍")
    assert turn.stopped == "max_tool_calls" and turn.steps == 1
    assert [e.status for e in turn.events] == ["ok", "ok", "skipped"]
    assert "已达到工具调用次数的上限（2 次）" in turn.answer
    assert [m.role for m in rig.session.messages[2:]] == ["assistant", "tool", "tool", "tool", "assistant"]


def test_token_limit(fixture_copy):
    big = Completion("", (TOP,), tokens_in=900, tokens_out=200, latency_ms=0.0, cached=False)
    rig = Rig(fixture_copy, [big, big, "不该走到这里"], max_tokens=2000)
    turn = rig.say("看一下")
    assert (turn.stopped, turn.steps) == ("max_tokens", 2)
    assert "已达到 token 用量的上限（2000）" in turn.answer


def test_time_limit(fixture_copy):
    ticks = iter(range(0, 1000, 40))  # 每看一次表过去 40 秒
    rig = Rig(fixture_copy, _endless(), max_seconds=100.0, max_steps=50)
    turn = rig.say("一直查", clock=lambda: float(next(ticks)))
    assert turn.stopped == "max_seconds" and turn.steps < 5
    assert "已达到时间上限（100 秒）" in turn.answer


# ---- 结果校验 ----


def test_a_made_up_number_gets_one_chance_to_be_fixed(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "成功率是 0.99。", "placeholder"])
    good = f"成功率是 {rig.top_rate()}。"
    rig.model._script[2] = good
    turn = rig.say("最高的成功率是多少？")

    assert (turn.answer, turn.retried, turn.steps) == (good, True, 3) and turn.verdict.ok
    nudge = rig.session.messages[-2]
    assert nudge.role == "user" and "0.99" in nudge.content and "找不到出处" in nudge.content
    assert rig.session.messages[-3].content == "成功率是 0.99。"  # 被退回的回答留在记录里
    assert not any("找不到出处" in s for s in rig.session.sources)  # 校验的提示不是用户说的话，不算出处
    # 模型重答时如果照抄被指出的数字，仍然判没有出处
    assert all("0.99" not in s for s in rig.session.sources)


def test_a_second_made_up_number_is_sent_with_a_warning(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "成功率是 0.99。", "成功率是 0.98。", "不该走到这里"])
    turn = rig.say("最高的成功率是多少？")
    assert (turn.retried, turn.steps) == (True, 3)
    assert turn.answer == "成功率是 0.98。\n\n（注意：以下数字没有在工具输出或你的输入里找到出处：0.98）"
    assert [m.text for m in turn.verdict.unsupported] == ["0.98"] and not turn.verdict.ok
    assert rig.session.messages[-1].content == turn.answer


def test_verification_can_be_turned_off_but_is_still_measured(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "成功率是 0.99。"], verify=False)
    turn = rig.say("最高的成功率是多少？")
    assert (turn.answer, turn.retried, turn.steps) == ("成功率是 0.99。", False, 2)
    assert [m.text for m in turn.verdict.unsupported] == ["0.99"]  # 关着也照样算，用来测量它的作用


def test_numbers_the_user_gave_are_supported(fixture_copy):
    rig = Rig(fixture_copy, ["好的，500 步。"])
    assert rig.say("训练 500 步要多久？").verdict.ok


# ---- 多轮 ----


def test_a_second_turn_builds_on_the_first(fixture_copy):
    rig = Rig(fixture_copy, [TOP, "查到了。", "placeholder"])
    rig.model._script[2] = f"刚才那个是 {rig.top_rate()}。"
    rig.say("谁最高？")
    turn = rig.say("它的成功率是多少？")
    assert turn.verdict.ok and turn.steps == 1  # 数字出自上一轮的工具输出
    assert [m.role for m in rig.model.requests[2].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert rig.session.sources[0] == "谁最高？" and rig.session.sources[-1] == "它的成功率是多少？"
    assert len(rig.session.sources) == 3


# ---- 上下文预算 ----


def test_long_observations_are_cut_for_the_model_but_kept_in_full_as_sources(fixture_copy):
    rig = Rig(fixture_copy, [ToolCall("c1", "describe_config", {}), "很多。"], observation_chars=300)
    turn = rig.say("都有哪些配置？")
    shown = rig.session.messages[3].content
    full = json.dumps(turn.events[0].observation, ensure_ascii=False)
    assert (
        len(full) > 2000
        and shown.startswith(full[:300])
        and shown.endswith("…（输出太长，后面的 " + str(len(full) - 300) + " 个字符已省略；请缩小查询范围）")
    )
    assert full in rig.session.sources


def test_older_tool_outputs_are_summarized_when_the_context_is_over_budget(fixture_copy):
    heavy = Completion(
        "", (ToolCall("c2", "list_policies", {}),), tokens_in=5000, tokens_out=5, latency_ms=0.0, cached=False
    )
    rig = Rig(
        fixture_copy,
        [ToolCall("c1", "leaderboard", {"suite": "bench"}), heavy, "好了。"],
        context_tokens=4000,
    )
    rig.say("先看排行榜再列策略")

    first, second, third = rig.model.requests
    assert second.messages[3].content == rig.session.messages[3].content  # 第二次请求时还没超预算
    # 第三次请求：上一次的输入超了预算，较早的那条工具输出换成一行摘要，最近一轮的原样保留
    assert (
        third.messages[3].content
        == '（较早的工具输出已省略：leaderboard {"suite": "bench"} 返回 7 条；需要时请重新调用）'
    )
    assert third.messages[5].content == rig.session.messages[5].content
    assert json.loads(rig.session.messages[3].content)["n_policies"] == 7  # 对话记录里原文还在


def test_every_tool_has_a_place_in_the_agent_prompt_or_specs():
    assert "需要确认" in AGENT_PROMPT and "工具" in AGENT_PROMPT
    assert len(tool_specs()) == len(TOOLS)
