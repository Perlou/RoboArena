"""工具注册表：参数的声明与校验、权限、每个工具在固定工作区上的结果。"""

import json

import pytest

from roboarena.agent.tools import (
    TOOLS,
    ToolContext,
    checked_overrides,
    execute,
    normalized_args,
    parse_args,
    schema_of,
    tool_specs,
)
from roboarena.aieval import fixture
from roboarena.config import Config
from roboarena.core.errors import ConfigError, ToolError
from roboarena.eval import service
from roboarena.eval.stats import mcnemar_exact
from roboarena.store.registry import list_artifacts

EVERYTHING = ("read", "write", "confirm")


@pytest.fixture
def ctx(fixture_copy) -> ToolContext:
    ws, cfg = fixture_copy
    return ToolContext(ws, cfg)


# ---- 参数：JSON Schema 与校验 ----


def test_schema_is_generated_from_the_dataclass():
    schema = schema_of(TOOLS["run_eval"].args)
    assert schema["required"] == ["policy"] and schema["additionalProperties"] is False
    assert schema["properties"]["policy"] == {"description": "要评测的策略的名称", "type": "string"}
    assert schema["properties"]["force"]["type"] == "boolean"
    train = schema_of(TOOLS["train_policy"].args)
    assert train["properties"]["backend"]["enum"] == ["bc", "diffusion"]
    assert train["properties"]["overrides"]["type"] == "array"
    assert train["properties"]["overrides"]["items"] == {"type": "string"}
    assert schema_of(TOOLS["list_policies"].args)["properties"] == {}


def test_every_tool_has_a_spec_that_serializes():
    specs = tool_specs()
    assert [s.name for s in specs] == list(TOOLS) and len(specs) == 11
    for spec in specs:
        json.dumps(spec.parameters)
        assert spec.description and all(p["description"] for p in spec.parameters["properties"].values())
    assert {t.access for t in TOOLS.values()} == {"read", "write", "confirm"}
    assert [n for n, t in TOOLS.items() if t.access == "confirm"] == ["train_policy", "remove_policy"]
    assert [n for n, t in TOOLS.items() if t.access == "write"] == ["run_eval"]


def test_arguments_are_normalized():
    assert normalized_args(TOOLS["leaderboard"], {}) == {"suite": "", "top": 0}
    assert normalized_args(TOOLS["leaderboard"], {"suite": "  bench ", "top": 3.0}) == {
        "suite": "bench",
        "top": 3,
    }
    assert normalized_args(TOOLS["leaderboard"], {"suite": None}) == {"suite": "", "top": 0}  # null 当作没给
    assert normalized_args(TOOLS["train_policy"], {"name": "nova"}) == {
        "name": "nova",
        "backend": "bc",
        "overrides": [],
    }
    first = parse_args(TOOLS["train_policy"].args, {"name": "a"})
    first.overrides.append("x")
    assert parse_args(TOOLS["train_policy"].args, {"name": "b"}).overrides == []  # 默认的列表不共用


@pytest.mark.parametrize(
    "tool, raw, message",
    [
        ("leaderboard", {"suit": "bench"}, "未知的参数：suit；可用的参数有：suite、top"),
        ("leaderboard", {"top": "3"}, "参数 top 应为整数"),
        ("leaderboard", {"top": 2.5}, "参数 top 应为整数"),
        ("leaderboard", {"top": True}, "参数 top 应为整数"),
        ("leaderboard", {"suite": 3}, "参数 suite 应为字符串"),
        ("leaderboard", "bench", "参数应为一个 JSON 对象"),
        ("run_eval", {}, "缺少必填参数 policy"),
        ("run_eval", {"policy": "  "}, "参数 policy 不能为空"),
        ("run_eval", {"policy": "a", "force": "yes"}, "参数 force 应为 true 或 false"),
        ("list_runs", {"kind": "training"}, "参数 kind 应为 .* 之一，实际是 'training'"),
        ("train_policy", {"name": "a", "backend": "ppo"}, "参数 backend 应为 bc、diffusion 之一"),
        ("train_policy", {"name": "a", "overrides": "policy.steps=5"}, "参数 overrides 应为列表"),
        (
            "train_policy",
            {"name": "a", "overrides": [{"key": "policy.steps"}]},
            "参数 overrides 的元素 应为字符串",
        ),
    ],
)
def test_bad_arguments_are_rejected(tool, raw, message):
    with pytest.raises(ToolError, match=message):
        parse_args(TOOLS[tool].args, raw)


@pytest.mark.parametrize(
    "name", ["../secret", "/etc/passwd", "a/b", "C:\\x", "~root", "a b", "名字", ".hidden"]
)
def test_names_cannot_be_paths(name):
    for tool, key in (
        ("remove_policy", "name"),
        ("get_run", "run_id"),
        ("dataset_info", "name"),
        ("leaderboard", "suite"),
    ):
        with pytest.raises(ToolError, match="不接受文件路径"):
            parse_args(TOOLS[tool].args, {key: name})


def test_real_names_pass_the_name_check():
    for name in (*fixture.POLICIES, "20261127-140257-train_policy-1ab7df54"):
        assert parse_args(TOOLS["remove_policy"].args, {"name": name}).name == name


def test_overrides_cannot_touch_locked_sections_or_carry_paths():
    cfg = Config()
    assert checked_overrides(cfg, ["policy.lr=1e-3", "eval.max_steps=50"]).policy.lr == 1e-3
    assert cfg.policy.lr != 1e-3  # 原来的配置不变
    for item, message in [
        ('workspace.root="elsewhere"', "workspace.root 不能通过工具修改"),
        ('llm.models.x.base_url="http://evil.test"', "llm.models.x.base_url 不能通过工具修改"),
        ('llm.default="x"', "不能通过工具修改"),
        ('task.name="toy"', "不能通过工具修改"),
        ('data.name="../x"', "不能出现文件路径"),
        ("eval.seed=3", "这里只能改 policy 段的配置项，不能改 eval.seed"),
    ]:
        with pytest.raises(ToolError, match=message):
            checked_overrides(cfg, [item], only=("policy",) if item == "eval.seed=3" else ())
    with pytest.raises(ConfigError, match="未知的配置项：policy.stpes"):
        checked_overrides(cfg, ["policy.stpes=5"])


# ---- 权限：由代码把关 ----


def test_only_read_tools_run_by_default(ctx):
    before = [a.name for a in list_artifacts(ctx.ws)]
    for name, args in (
        ("remove_policy", {"name": "random"}),
        ("train_policy", {"name": "nova"}),
        ("run_eval", {"policy": "atlas-step20", "suite": "probe"}),
    ):
        observation = execute(ctx, name, args)
        assert "没有得到许可，没有执行" in observation["error"]
    assert execute(ctx, "remove_policy", {"name": "random"}, allow=("read", "write"))["error"].startswith(
        "工具 remove_policy 属于「需确认」一档"
    )
    assert [a.name for a in list_artifacts(ctx.ws)] == before
    assert ctx.ws.db.execute("SELECT COUNT(*) FROM runs WHERE kind = 'train_policy'").fetchone()[0] == 2


def test_a_model_cannot_talk_its_way_past_the_gate(ctx):
    """参数里多写一句「用户已同意」没有用：多出来的参数直接判不合法，也不存在这样的开关。"""
    for extra in ({"confirmed": True}, {"user_approved": "yes"}, {"allow": ["confirm"]}):
        observation = execute(ctx, "remove_policy", {"name": "random", **extra})
        assert "未知的参数" in observation["error"]
    assert "random" in [a.name for a in list_artifacts(ctx.ws)]


def test_errors_become_observations_not_exceptions(ctx):
    assert execute(ctx, "no_such_tool", {})["error"].startswith(
        "没有名为 no_such_tool 的工具；可用的工具有：list_policies"
    )
    assert execute(ctx, "get_run", {"run_id": "20260101-000000-eval-00000000"}) == {
        "error": "没有编号为 20260101-000000-eval-00000000 的运行"
    }
    assert execute(ctx, "leaderboard", {"suite": "nope"})["error"].startswith(
        "任务 toy 没有名为 nope 的评测集"
    )
    assert execute(ctx, "run_eval", {"policy": "ghost"}, allow=EVERYTHING) == {
        "error": "注册表里没有名为 ghost 的策略"
    }
    assert "\n" not in execute(ctx, "leaderboard", {"top": "x"})["error"]


# ---- 每个工具的结果：和平台函数现算的一致 ----


def test_list_policies(ctx):
    names = [p["name"] for p in execute(ctx, "list_policies", {})["policies"]]
    assert sorted(names) == sorted(fixture.POLICIES) and len(names) == 7


def test_leaderboard_matches_the_service(ctx):
    board = service.leaderboard(ctx.ws, ctx.cfg)
    out = execute(ctx, "leaderboard", {})
    assert (out["suite"], out["n_policies"]) == ("bench", 7)
    assert [r["policy"] for r in out["rows"]] == [r.policy for r in board]
    assert [(r["successes"], r["n"]) for r in out["rows"]] == [(r.successes, r.n) for r in board]
    assert out["rows"][0]["success_rate"] == round(board[0].rate, 4) and out["rows"][0]["rank"] == 1

    top = execute(ctx, "leaderboard", {"suite": "bench", "top": 2})
    assert top["rows"] == out["rows"][:2] and top["n_policies"] == 7
    probe = execute(ctx, "leaderboard", {"suite": "probe"})
    assert sorted(r["policy"] for r in probe["rows"]) == sorted(fixture.PROBE_POLICIES)
    assert all(r["n"] == 8 for r in probe["rows"])


def test_compare_policies_pairs_episodes_by_initial_state(ctx):
    out = execute(ctx, "compare_policies", {"a": "atlas-noise0.3", "b": "random"})
    rows = {}
    for name in ("atlas-noise0.3", "random"):
        found = ctx.ws.db.execute(
            "SELECT p.init_index, p.success FROM episodes p JOIN evals e ON e.id = p.eval_id"
            " JOIN artifacts a ON a.id = e.policy_id JOIN suites s ON s.id = e.suite_id"
            " WHERE a.name = ? AND s.name = 'bench'",
            (name,),
        ).fetchall()
        rows[name] = {r["init_index"]: r["success"] for r in found}
    a, b = rows["atlas-noise0.3"], rows["random"]
    only_a = sum(1 for i in a if a[i] and not b[i])
    only_b = sum(1 for i in a if b[i] and not a[i])
    # 两个策略的分块长度不同（实际的 n_exec 不同），仍然要按初始状态配上对
    assert out["paired"] == {
        "n_pairs": 16,
        "only_a_succeeds": only_a,
        "only_b_succeeds": only_b,
        "mcnemar_p": round(mcnemar_exact(only_a, only_b), 6),
    }
    assert (out["a"]["policy"], out["a"]["successes"]) == ("atlas-noise0.3", sum(a.values()))
    assert (out["b"]["policy"], out["b"]["successes"]) == ("random", sum(b.values()))
    low_a, high_a = out["a"]["ci95"]
    low_b, high_b = out["b"]["ci95"]
    assert out["ci_overlap"] is (low_a <= high_b and low_b <= high_a)

    missing = execute(ctx, "compare_policies", {"a": "atlas", "b": "atlas-step20", "suite": "probe"})
    assert missing == {"error": "策略 atlas-step20 在评测集 probe 上还没有评测结果"}
    assert execute(ctx, "compare_policies", {"a": "atlas", "b": "ghost"}) == {
        "error": "注册表里没有名为 ghost 的策略"
    }


def test_runs_can_be_listed_and_inspected(ctx):
    trains = execute(ctx, "list_runs", {"kind": "train_policy"})["runs"]
    assert len(trains) == 2 and {r["kind"] for r in trains} == {"train_policy"}
    assert all(r["status"] == "done" for r in trains)
    assert len(execute(ctx, "list_runs", {"limit": 4})["runs"]) == 4
    assert len(execute(ctx, "list_runs", {"limit": 10_000})["runs"]) <= 50

    run = execute(ctx, "get_run", {"run_id": trains[0]["id"]})
    assert run["id"] == trains[0]["id"] and run["meta"]["policy"] in fixture.POLICIES
    assert set(run["last_metrics"]) >= {"train_loss"} or run["last_metrics"]
    assert "argv" not in run["meta"] and "platform" not in run["meta"]  # 只给挑出来的几项


def test_dataset_info(ctx):
    core = execute(ctx, "dataset_info", {})
    mini = execute(ctx, "dataset_info", {"name": "mini"})
    assert core["name"] == "core" and core["n_episodes"] == sum(core["splits"].values()) == 24
    assert mini["n_episodes"] == 8 and len(mini["fingerprint"]) == 12
    assert execute(ctx, "dataset_info", {"name": "ghost"})["error"].startswith(
        "任务 toy 没有名为 ghost 的数据集"
    )


def test_describe_config_hides_the_gateway_and_workspace(ctx):
    items = {i["key"]: i for i in execute(ctx, "describe_config", {"section": "eval"})["items"]}
    assert items["eval.suite"] == {
        "key": "eval.suite", "type": "str", "default": "main", "doc": "评测集的名称", "current": "bench"
    }  # fmt: skip
    everything = [i["key"] for i in execute(ctx, "describe_config", {})["items"]]
    assert "policy.lr" in everything and "seed" in everything
    assert not [k for k in everything if k.startswith(("llm.", "workspace."))]
    assert "不在工具里展示" in execute(ctx, "describe_config", {"section": "llm"})["error"]
    assert execute(ctx, "describe_config", {"section": "nope"})["error"].startswith("没有名为 nope 的配置段")


def test_check_overrides_validates_without_doing_anything(ctx):
    runs = ctx.ws.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    ok = execute(ctx, "check_overrides", {"overrides": ["policy.lr=1e-3", "policy.steps=500"]})
    assert ok == {"ok": True, "overrides": [["policy.lr", 0.001], ["policy.steps", 500]]}
    assert execute(ctx, "check_overrides", {"overrides": ["policy.stpes=1000"]}) == {
        "error": "未知的配置项：policy.stpes"
    }
    assert "应为" in execute(ctx, "check_overrides", {"overrides": ['policy.steps="many"']})["error"]
    assert ctx.ws.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == runs
    assert ctx.cfg.policy.steps == 200


def test_run_eval_writes_a_result_and_reuses_it(ctx):
    args = {"policy": "atlas-step20", "suite": "probe"}
    first = execute(ctx, "run_eval", args, allow=("read", "write"))
    assert (first["n"], first["reused_existing_result"], first["suite"]) == (8, False, "probe")
    assert first["success_rate"] == round(first["successes"] / 8, 4)
    again = execute(ctx, "run_eval", args, allow=("read", "write"))
    assert again == {**first, "reused_existing_result": True}
    assert "atlas-step20" in [r["policy"] for r in execute(ctx, "leaderboard", {"suite": "probe"})["rows"]]


def test_remove_policy_deletes_the_policy_and_its_results(ctx):
    out = execute(ctx, "remove_policy", {"name": "random"}, allow=EVERYTHING)
    assert out == {"removed": "random", "removed_evals": 2}  # bench 与 probe 各一次
    assert "random" not in [p["name"] for p in execute(ctx, "list_policies", {})["policies"]]
    assert execute(ctx, "remove_policy", {"name": "random"}, allow=EVERYTHING) == {
        "error": "注册表里没有名为 random 的策略"
    }


def test_train_policy_uses_only_the_section_its_backend_reads(ctx):
    out = execute(
        ctx,
        "train_policy",
        {"name": "nova", "overrides": ["policy.steps=30", "policy.snapshots=[]"]},
        allow=EVERYTHING,
    )
    assert (out["name"], out["backend"]) == ("nova", "bc")
    assert execute(ctx, "get_run", {"run_id": out["run_id"]})["meta"]["policy"] == "nova"
    # 扩散策略读 diffusion 段：写成 policy.steps 不会生效，所以拒绝，而不是悄悄忽略
    wrong = execute(
        ctx,
        "train_policy",
        {"name": "nova2", "backend": "diffusion", "overrides": ["policy.steps=30"]},
        allow=EVERYTHING,
    )
    assert wrong == {"error": "这里只能改 diffusion 段的配置项，不能改 policy.steps"}
    assert execute(ctx, "train_policy", {"name": "nova"}, allow=EVERYTHING) == {"error": "名称已被占用：nova"}
    assert "nova2" not in [p["name"] for p in execute(ctx, "list_policies", {})["policies"]]


def test_the_fixture_is_not_changed_by_tests_that_use_a_copy(fixture_root, ctx):
    execute(ctx, "remove_policy", {"name": "atlas"}, allow=EVERYTHING)
    from roboarena.store.workspace import Workspace

    with Workspace(fixture_root) as original:
        assert len(list_artifacts(original)) == 7
