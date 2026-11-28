"""对话的记忆：一次对话是一条 chat 类型的运行记录，可以接着上次继续（FR-G5）。"""

import json

import pytest

from roboarena.agent.chat import open_chat
from roboarena.agent.loop import Session
from roboarena.agent.tools import execute
from roboarena.core.errors import ConflictError, ModelError, NotFoundError, RoboArenaError
from roboarena.llm.api import Message, ToolCall
from roboarena.llm.gateway import Gateway, read_calls
from roboarena.llm.scripted import ScriptedModel
from roboarena.store.runs import resume_run, start_run

TOP = ToolCall("c1", "leaderboard", {"top": 1})


def _gateway(cfg, script) -> tuple[Gateway, ScriptedModel]:
    model = ScriptedModel(cfg.llm.default, script)
    return Gateway(cfg.llm, models={cfg.llm.default: model}), model


def _lines(path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_session_round_trips_through_json():
    session = Session.new("系统提示")
    session.messages += [
        Message("user", "谁最高？"),
        Message(
            "assistant",
            "",
            tool_calls=(ToolCall("c1", "leaderboard", {"top": 1}), ToolCall("c2", "x", {}, "[1]")),
        ),
        Message("tool", "{}", tool_call_id="c1"),
    ]
    session.sources += ["谁最高？", "{}"]
    session.summaries[3] = "（摘要）"
    session.compact_before, session.last_tokens_in = 2, 777
    again = Session.from_dict(json.loads(json.dumps(session.to_dict(), ensure_ascii=False)))
    assert again == session and again.summaries == {3: "（摘要）"}


def test_a_chat_is_a_run_that_keeps_its_transcript(fixture_copy):
    ws, cfg = fixture_copy
    gateway, _ = _gateway(cfg, [TOP, "查到了。"])
    with open_chat(ws, cfg, gateway) as chat:
        turn = chat.say("谁最高？")
        run_dir = chat.run.dir
        assert gateway.calls_path == run_dir / "llm_calls.jsonl"

    assert turn.answer == "查到了。" and gateway.calls_path is None
    row = ws.db.execute("SELECT kind, status FROM runs WHERE id = ?", (chat.run.id,)).fetchone()
    assert (row["kind"], row["status"]) == ("chat", "done")
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert (meta["model"], meta["turns"]) == (cfg.llm.default, 1)
    assert len(read_calls(run_dir / "llm_calls.jsonl")) == 2

    (record,) = _lines(run_dir / "turns.jsonl")
    assert (record["user"], record["answer"], record["stopped"], record["steps"]) == (
        "谁最高？",
        "查到了。",
        "answer",
        2,
    )
    assert record["events"][0]["name"] == "leaderboard" and record["events"][0]["status"] == "ok"
    assert record["events"][0]["observation"] == execute(chat.ctx, "leaderboard", {"top": 1})
    assert (record["numbers"], record["unsupported"], record["retried"]) == (0, [], False)

    saved = Session.from_dict(json.loads((run_dir / "session.json").read_text(encoding="utf-8")))
    assert [m.role for m in saved.messages] == ["system", "user", "assistant", "tool", "assistant"]


def test_resume_continues_the_same_conversation(fixture_copy):
    ws, cfg = fixture_copy
    gateway, _ = _gateway(cfg, [TOP, "查到了。"])
    with open_chat(ws, cfg, gateway) as chat:
        chat.say("谁最高？")
    rate = execute(chat.ctx, "leaderboard", {"top": 1})["rows"][0]["success_rate"]

    gateway, model = _gateway(cfg, [f"它的成功率是 {rate}。"])
    with open_chat(ws, cfg, gateway, resume=chat.run.id) as again:
        turn = again.say("它的成功率是多少？")

    assert again.run.id == chat.run.id and turn.verdict.ok  # 数字出自上一次的工具输出
    assert [m.role for m in model.requests[0].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert ws.db.execute("SELECT COUNT(*) FROM runs WHERE kind = 'chat'").fetchone()[0] == 1
    assert ws.db.execute("SELECT status FROM runs WHERE id = ?", (chat.run.id,)).fetchone()[0] == "done"
    assert [r["user"] for r in _lines(chat.run.dir / "turns.jsonl")] == ["谁最高？", "它的成功率是多少？"]
    assert json.loads((chat.run.dir / "meta.json").read_text(encoding="utf-8"))["turns"] == 2
    assert len(read_calls(chat.run.dir / "llm_calls.jsonl")) == 3  # 接着记在同一个文件里


def test_resume_uses_the_model_of_the_first_conversation_unless_told_otherwise(fixture_copy):
    ws, cfg = fixture_copy
    other = next(name for name in cfg.llm.models if name != cfg.llm.default)
    models = {name: ScriptedModel(name, [f"我是 {name}"] * 3) for name in cfg.llm.models}
    gateway = Gateway(cfg.llm, models=models)
    with open_chat(ws, cfg, gateway, model=other) as chat:
        assert chat.say("你是谁").answer == f"我是 {other}"
    with open_chat(ws, cfg, gateway, resume=chat.run.id) as again:
        assert again.say("再说一遍").answer == f"我是 {other}"
    with open_chat(ws, cfg, gateway, resume=chat.run.id, model=cfg.llm.default) as switched:
        assert switched.say("换一个").answer == f"我是 {cfg.llm.default}"


def test_only_chats_can_be_resumed(fixture_copy):
    ws, cfg = fixture_copy
    gateway, _ = _gateway(cfg, [])
    eval_run = ws.db.execute("SELECT id FROM runs WHERE kind = 'eval' LIMIT 1").fetchone()[0]
    with pytest.raises(RoboArenaError, match=f"运行 {eval_run} 不是一次对话（类型是 eval）"):
        with open_chat(ws, cfg, gateway, resume=eval_run):
            pass
    with pytest.raises(NotFoundError, match="没有编号为 nope 的运行"):
        with open_chat(ws, cfg, gateway, resume="nope"):
            pass
    assert (
        ws.db.execute("SELECT status FROM runs WHERE id = ?", (eval_run,)).fetchone()[0] == "done"
    )  # 没被动过


def test_a_failed_turn_does_not_damage_the_earlier_ones(fixture_copy):
    ws, cfg = fixture_copy
    gateway, _ = _gateway(cfg, ["第一句的回答。", TOP])  # 第二轮：调完工具之后脚本用完，模型报错
    with pytest.raises(ModelError):
        with open_chat(ws, cfg, gateway) as chat:
            chat.say("第一句")
            chat.say("第二句")
    assert ws.db.execute("SELECT status FROM runs WHERE id = ?", (chat.run.id,)).fetchone()[0] == "failed"
    assert [r["user"] for r in _lines(chat.run.dir / "turns.jsonl")] == ["第一句"]

    gateway, model = _gateway(cfg, ["接上了。"])
    with open_chat(ws, cfg, gateway, resume=chat.run.id) as again:
        again.say("第二句，再来")
    # 出错的那半轮（用户的话、调了一半的工具）不在记录里
    assert [m.content for m in model.requests[0].messages[1:]] == ["第一句", "第一句的回答。", "第二句，再来"]


def test_done_runs_can_only_be_reopened_on_request(ws):
    with start_run(ws, "chat", config={}, config_toml="", seed=0) as run:
        pass
    with pytest.raises(ConflictError, match="已经完成"):
        with resume_run(ws, run.id):
            pass
    with resume_run(ws, run.id, allow_done=True) as again:
        assert again.id == run.id
        assert ws.db.execute("SELECT status FROM runs WHERE id = ?", (run.id,)).fetchone()[0] == "running"
    assert ws.db.execute("SELECT status FROM runs WHERE id = ?", (run.id,)).fetchone()[0] == "done"
