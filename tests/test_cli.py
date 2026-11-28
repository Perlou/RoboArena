import pytest

from roboarena.cli import main

from .conftest import QUICK


@pytest.fixture
def run(tmp_path, capsys):
    base = ["--config", str(QUICK), "--set", f'workspace.root="{tmp_path / "ws"}"']

    def _run(*argv):
        code = main([*argv, *base])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


def test_full_flow(run):
    code, out, _ = run("info")
    assert code == 0 and "当前任务：toy" in out and "评测集：0" in out

    code, out, _ = run("data", "suite")
    assert code == 0 and "16 个初始状态" in out
    code, _, err = run("data", "suite")
    assert code == 1 and "错误：" in err and "已有名为 main 的评测集" in err

    assert run("policy", "add", "--name", "seek", "--backend", "toy_seek")[0] == 0
    assert run("policy", "add", "--name", "random", "--backend", "random", "--params", '{"chunk": 2}')[0] == 0
    code, out, _ = run("policy", "list")
    assert code == 0 and "seek" in out and "toy_seek" in out and "random" in out

    code, out, _ = run("eval", "run", "--policy", "seek", "random")
    assert code == 0 and "seek @ real：16/16 成功" in out and "未重跑" not in out
    code, out, _ = run("eval", "run", "--policy", "seek")
    assert code == 0 and "未重跑" in out

    code, out, _ = run("eval", "leaderboard")
    lines = out.splitlines()
    assert code == 0 and "95% Wilson 区间" in lines[1]
    assert lines[2].split()[:4] == ["1", "seek", "16/16", "1.000"]
    assert lines[3].split()[1] == "random"

    code, out, _ = run("runs", "list")
    assert code == 0 and out.count("  eval  ") == 2 and "done" in out


def test_errors_are_reported_in_one_line(run):
    code, _, err = run("eval", "run", "--policy", "ghost")
    assert code == 1 and err.strip().startswith("错误：") and "Traceback" not in err
    code, out, _ = run("eval", "leaderboard")
    assert code == 1  # 评测集还不存在

    assert main(["info", "--set", "nope=1"]) == 1


def test_empty_listings(run):
    assert "还没有登记任何策略" in run("policy", "list")[1]
    assert "还没有运行记录" in run("runs", "list")[1]
    run("data", "suite")
    assert "还没有评测结果" in run("eval", "leaderboard")[1]


def test_task_list(run):
    code, out, _ = run("task", "list")
    assert code == 0 and "\ntoy：" in out and '"max_steps": 30' in out and out.startswith("pusht：")


@pytest.mark.parametrize("command", ["report", "pipeline", "bench", "serve", "token"])
def test_planned_commands_say_so(run, command):
    code, _, err = run(command)
    assert code == 2 and "尚未实现" in err


def test_bad_policy_params(run):
    code, _, err = run("policy", "add", "--name", "x", "--backend", "random", "--params", "[1]")
    assert code == 1 and "JSON 对象" in err


# ---- 应用层的命令：模型用脚本化假模型顶替 ----

from roboarena import cli  # noqa: E402
from roboarena.core.errors import ModelError  # noqa: E402
from roboarena.llm.api import ToolCall  # noqa: E402
from roboarena.llm.gateway import Gateway  # noqa: E402
from roboarena.llm.scripted import ScriptedModel  # noqa: E402


@pytest.fixture
def scripted(monkeypatch):
    """让命令行用假模型：scripted(名称=回答列表, …)。返回的字典里是建出来的假模型。"""
    made: dict[str, ScriptedModel] = {}

    def use(**scripts):
        made.clear()  # 同一个测试里重新指定脚本时，从头开始

        def make_gateway(cfg, ws):
            for name, script in scripts.items():
                made.setdefault(name, ScriptedModel(name, script))
            return Gateway(cfg.llm, models=dict(made))

        monkeypatch.setattr(cli, "make_gateway", make_gateway)
        return made

    return use


def test_llm_list_shows_the_configured_models(run):
    code, out, _ = run("llm", "list")
    lines = out.splitlines()
    assert code == 0 and lines[0].split() == ["名称", "后端", "模型", "服务地址", "默认"]
    assert lines[1].split() == ["qwen7b", "ollama", "qwen2.5:7b", "http://127.0.0.1:11434", "是"]
    assert lines[2].split()[:3] == ["qwen08b", "ollama", "qwen3.5:0.8b"]


def test_llm_ping_reports_each_model_and_fails_if_any_does(run, scripted):
    def broken(messages, tools):
        raise ModelError("连不上模型 qwen08b 的服务", retryable=False)

    scripted(qwen7b=["好"], qwen08b=broken)
    code, out, _ = run("llm", "ping")
    assert code == 1
    assert "qwen7b：正常" in out and "毫秒" in out
    assert "qwen08b：失败：连不上模型 qwen08b 的服务" in out

    scripted(qwen7b=["好"])
    code, out, _ = run("llm", "ping", "--model", "qwen7b")
    assert code == 0 and "qwen08b" not in out


def test_agent_tools_lists_tools_with_access_and_parameters(run):
    code, out, _ = run("agent", "tools")
    lines = out.splitlines()
    assert code == 0 and len(lines) == 12  # 表头 + 11 个工具
    assert lines[0].split() == ["工具", "权限", "参数", "说明"]
    removal = next(line for line in lines if line.startswith("remove_policy"))
    assert removal.split()[1:3] == ["需确认", "name*"]
    board = next(line for line in lines if line.startswith("leaderboard"))
    assert board.split()[1:3] == ["只读", "suite、top"]


def test_chat_once_answers_and_leaves_a_run(run, scripted):
    scripted(qwen7b=[ToolCall("c1", "list_policies", {}), "还没有登记任何策略。", "刚才说过了，没有。"])
    code, out, _ = run("chat", "--once", "有哪些策略？")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "· list_policies {} → 成功"
    assert lines[1] == "还没有登记任何策略。"
    assert lines[2].startswith("（对话 ") and "模型调用 2 次" in lines[2]
    run_id = lines[2].split()[1]
    assert "-chat-" in run_id

    code, out, _ = run("chat", "--once", "再说一遍", "--resume", run_id)
    assert code == 0 and out.splitlines()[0] == "刚才说过了，没有。" and run_id in out
    code, out, _ = run("runs", "list")
    assert out.count("  chat  ") == 1


def test_chat_once_does_not_run_confirm_tools_without_yes(run, scripted):
    assert run("policy", "add", "--name", "random", "--backend", "random")[0] == 0
    removal = ToolCall("c1", "remove_policy", {"name": "random"})
    scripted(qwen7b=[removal, "需要你确认，没有删除。", removal, "已删除。"])

    code, out, _ = run("chat", "--once", "把 random 删掉")
    assert code == 0 and '· remove_policy {"name": "random"} → 没有得到确认，未执行' in out
    assert "random" in run("policy", "list")[1]

    code, out, _ = run("chat", "--once", "把 random 删掉", "--yes")
    assert code == 0 and '· remove_policy {"name": "random"} → 成功（已预先同意）' in out
    assert "还没有登记任何策略" in run("policy", "list")[1]


def test_interactive_chat_asks_before_confirm_tools(run, scripted, monkeypatch):
    assert run("policy", "add", "--name", "random", "--backend", "random")[0] == 0
    removal = ToolCall("c1", "remove_policy", {"name": "random"})
    scripted(qwen7b=[removal, "没有删除。", removal, "已删除。"])
    typed = iter(["把 random 删掉", "n", "", "还是删掉吧", "y", "exit"])
    prompts: list[str] = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return next(typed)

    monkeypatch.setattr("builtins.input", fake_input)
    code, out, _ = run("chat")
    assert code == 0 and "没有删除。" in out and "已删除。" in out
    assert prompts.count("你> ") == 4  # 两句话、一个空行、exit
    assert [p for p in prompts if p != "你> "] == ['确认执行 remove_policy {"name": "random"}？[y/N] '] * 2
    assert "用户没有同意，未执行" in out and "成功（用户已确认）" in out
    assert "还没有登记任何策略" in run("policy", "list")[1]
    assert run("runs", "list")[1].count("  chat  ") == 1  # 整段交互是一次对话


def test_interactive_chat_ends_at_end_of_input(run, scripted, monkeypatch):
    scripted(qwen7b=["你好。"])
    typed = iter(["你好"])

    def fake_input(prompt=""):
        try:
            return next(typed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    code, out, _ = run("chat")
    assert code == 0 and "你好。" in out


def test_chat_reports_a_model_failure_in_one_line(run, scripted):
    scripted(qwen7b=[])
    code, _, err = run("chat", "--once", "在吗")
    assert code == 1 and err.count("\n") == 1 and "错误：" in err and "脚本只有 0 条回答" in err


def test_aieval_task_suite_cli_flow(run, scripted, fixture_root, monkeypatch):
    from roboarena.aieval import run as aieval_run

    monkeypatch.setattr(aieval_run, "ensure_fixture", lambda cache: fixture_root)

    code, out, _ = run("aieval", "report", "--suite", "task")
    assert code == 0 and "还没有评测结果" in out

    code, _, err = run("aieval", "run", "--suite", "task", "--bucket", "seen")
    assert code == 1 and "task 套件不分桶" in err

    scripted(qwen7b=["回答。" for _ in range(10)])
    code, out, _ = run("aieval", "run", "--suite", "task", "--seeds", "0", "--limit", "2")
    assert code == 0 and "qwen7b（verify/ollama-" in out and "/2，" in out

    code, out, _ = run("aieval", "report", "--suite", "task")
    assert code == 0 and "# AI 评测报告：task" in out and "任务完成率" in out
