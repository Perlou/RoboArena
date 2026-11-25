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
