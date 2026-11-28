"""工具注册表（docs/ARCHITECTURE.md 16.3 节）：把平台已有的操作包装成模型可以调用的工具。

参数用 dataclass 声明：由它生成给模型看的 JSON Schema，也由它校验模型给的参数。
工具只接受名称与数值，不接受文件路径。没通过校验或没有得到许可的调用不会执行。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Literal, get_args, get_origin, get_type_hints

from roboarena.config import Config, describe_config, split_override, with_overrides
from roboarena.core.errors import NotFoundError, RoboArenaError, ToolError
from roboarena.data.dataset import load_dataset
from roboarena.eval import service
from roboarena.eval.stats import mcnemar_exact
from roboarena.llm.api import ToolSpec
from roboarena.policies.train import train_policy
from roboarena.store import evals as eval_store
from roboarena.store.registry import delete_artifact, get_artifact, list_artifacts
from roboarena.store.runs import read_metrics
from roboarena.store.suites import load_suite
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task

Access = Literal["read", "write", "confirm"]
ACCESS_ZH = {"read": "只读", "write": "可写", "confirm": "需确认"}

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")
# 这几段配置不能经由工具修改：工作区的位置、模型服务的地址、当前任务
LOCKED_SECTIONS = ("workspace", "llm", "task")
MAX_LISTED_RUNS = 50


@dataclass(frozen=True)
class ToolContext:
    ws: Workspace
    cfg: Config


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type  # 参数的 dataclass
    access: Access
    run: Callable[[ToolContext, Any], dict[str, Any]]  # 返回值是给模型看的观察

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, schema_of(self.args))


def arg(doc: str, default: Any = MISSING, *, name: bool = False, choices: tuple[str, ...] = ()) -> Any:
    """声明一个参数。name=True 表示它是平台里某样东西的名称（按名称的字符集校验，所以不可能是路径）。"""
    meta = {"doc": doc, "name": name, "choices": choices}
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata=meta)
    return field(default=default, metadata=meta)


# ---- 参数：生成 JSON Schema、校验 ----

_JSON_TYPES: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def schema_of(args: type) -> dict[str, Any]:
    hints = get_type_hints(args)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in fields(args):
        hint = hints[f.name]
        prop: dict[str, Any] = {"description": f.metadata["doc"]}
        if get_origin(hint) is list:
            prop |= {"type": "array", "items": {"type": _JSON_TYPES[get_args(hint)[0]]}}
        else:
            prop["type"] = _JSON_TYPES[hint]
        if f.metadata["choices"]:
            prop["enum"] = list(f.metadata["choices"])
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)
        properties[f.name] = prop
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _check_value(value: Any, hint: Any, key: str, meta: Mapping[str, Any]) -> Any:
    if get_origin(hint) is list:
        if not isinstance(value, list):
            raise ToolError(f"参数 {key} 应为列表")
        return [_check_value(v, get_args(hint)[0], f"{key} 的元素", meta) for v in value]
    if hint is bool:
        if not isinstance(value, bool):
            raise ToolError(f"参数 {key} 应为 true 或 false")
        return value
    if hint is int:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolError(f"参数 {key} 应为整数")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolError(f"参数 {key} 应为数值")
        return float(value)
    if not isinstance(value, str):
        raise ToolError(f"参数 {key} 应为字符串")
    value = value.strip()
    if meta["choices"] and value not in meta["choices"]:
        raise ToolError(f"参数 {key} 应为 {'、'.join(meta['choices'])} 之一，实际是 {value!r}")
    if meta["name"] and value and not _NAME.fullmatch(value):
        raise ToolError(f"参数 {key} 应为名称（字母、数字与 . _ : -），不接受文件路径：{value!r}")
    return value


def parse_args(args: type, raw: Any) -> Any:
    """按参数声明校验模型给的参数，返回 args 的实例。省略或为 null 的可选参数取默认值。"""
    if not isinstance(raw, Mapping):
        raise ToolError("工具的参数应为一个 JSON 对象")
    hints = get_type_hints(args)
    declared = {f.name: f for f in fields(args)}
    if unknown := [k for k in raw if k not in declared]:
        raise ToolError(
            f"未知的参数：{'、'.join(map(str, unknown))}；可用的参数有：{'、'.join(declared) or '（无）'}"
        )
    values: dict[str, Any] = {}
    for name, f in declared.items():
        required = f.default is MISSING and f.default_factory is MISSING
        if raw.get(name) is None:
            if required:
                raise ToolError(f"缺少必填参数 {name}")
            continue
        values[name] = _check_value(raw[name], hints[name], name, f.metadata)
        if required and values[name] == "":
            raise ToolError(f"参数 {name} 不能为空")
    return args(**values)


def checked_overrides(cfg: Config, overrides: list[str], only: tuple[str, ...] = ()) -> Config:
    """校验一组配置覆盖项并返回应用之后的配置。only 给出时只许改这几段。"""
    for item in overrides:
        key = split_override(item)[0]
        section = key.split(".")[0]
        if section in LOCKED_SECTIONS:
            raise ToolError(f"配置项 {key} 不能通过工具修改")
        if only and section not in only:
            raise ToolError(f"这里只能改 {'、'.join(only)} 段的配置项，不能改 {key}")
        if "/" in item or "\\" in item:
            raise ToolError(f"覆盖项里不能出现文件路径：{item!r}")
    return with_overrides(cfg, overrides)


# ---- 第一批工具 ----


@dataclass(frozen=True)
class NoArgs:
    pass


@dataclass(frozen=True)
class LeaderboardArgs:
    suite: str = arg("评测集的名称；省略时用当前配置的评测集", "", name=True)
    top: int = arg("只要前几名；0 表示全部", 0)


@dataclass(frozen=True)
class CompareArgs:
    a: str = arg("第一个策略的名称", name=True)
    b: str = arg("第二个策略的名称", name=True)
    suite: str = arg("评测集的名称；省略时用当前配置的评测集", "", name=True)


RUN_KINDS = ("data", "train_policy", "eval", "chat", "aieval")


@dataclass(frozen=True)
class ListRunsArgs:
    kind: str = arg(
        "只列某一类运行：data 数据采集与构建、train_policy 策略训练、eval 评测、chat 对话、aieval AI 评测；省略表示全部",
        "",
        choices=("", *RUN_KINDS),
    )
    limit: int = arg("最多列几条，从最近的开始", 10)


@dataclass(frozen=True)
class GetRunArgs:
    run_id: str = arg("运行编号，形如 20261126-005938-eval-0e96ec17", name=True)


@dataclass(frozen=True)
class DatasetArgs:
    name: str = arg("数据集的名称；省略时用当前配置的数据集", "", name=True)


@dataclass(frozen=True)
class DescribeConfigArgs:
    section: str = arg("配置段的名称，如 policy、eval、data；省略表示全部", "", name=True)


_OVERRIDES_DOC = (
    "配置覆盖项，每项形如 键=值。常用的键：policy.steps 训练步数、policy.lr 学习率、policy.batch 批大小、"
    "policy.n_demos 演示条数、eval.max_steps 每回合步数上限；其余的键用 describe_config 查"
)


@dataclass(frozen=True)
class CheckOverridesArgs:
    overrides: list[str] = arg(_OVERRIDES_DOC)


@dataclass(frozen=True)
class RunEvalArgs:
    policy: str = arg("要评测的策略的名称", name=True)
    suite: str = arg("评测集的名称；省略时用当前配置的评测集", "", name=True)
    force: bool = arg("已有结果时是否删掉重跑", False)


@dataclass(frozen=True)
class TrainPolicyArgs:
    name: str = arg("新策略登记用的名称", name=True)
    backend: str = arg(
        "算法：bc 行为克隆，diffusion 扩散策略（Diffusion Policy）", "bc", choices=("bc", "diffusion")
    )
    overrides: list[str] = arg(
        "训练参数的覆盖项，每项形如 键=值。backend 为 bc 时写 policy 段的键（policy.steps 训练步数、policy.lr 学习率、"
        "policy.batch 批大小、policy.n_demos 演示条数）；backend 为 diffusion 时写 diffusion 段里同名的键"
        "（diffusion.steps、diffusion.lr、diffusion.batch 等）",
        [],
    )


@dataclass(frozen=True)
class RemovePolicyArgs:
    name: str = arg("要删除的策略的名称", name=True)


TRAIN_SECTIONS = {"bc": "policy", "diffusion": "diffusion"}  # 算法 → 它读的配置段


def _task_name(ctx: ToolContext) -> str:
    return get_task(ctx.cfg.task.name).spec.name


def _with_suite(cfg: Config, suite: str) -> Config:
    return with_overrides(cfg, [f'eval.suite="{suite}"']) if suite else cfg


def _board_row(rank: int, r: service.LeaderboardRow) -> dict[str, Any]:
    return {
        "rank": rank,
        "policy": r.policy,
        "successes": r.successes,
        "n": r.n,
        "success_rate": round(r.rate, 4),
        "ci95": [round(r.ci_low, 4), round(r.ci_high, 4)],
        "mean_max_progress": round(r.mean_max_progress, 4),
    }


def _list_policies(ctx: ToolContext, args: NoArgs) -> dict[str, Any]:
    items = list_artifacts(ctx.ws, task=_task_name(ctx), kind="policy")
    listed = [{"name": a.name, "backend": a.backend, "created_at": a.created_at} for a in items]
    return {"n_policies": len(listed), "policies": listed}  # 个数直接给出来：模型自己数出来的数没有出处


def _leaderboard(ctx: ToolContext, args: LeaderboardArgs) -> dict[str, Any]:
    cfg = _with_suite(ctx.cfg, args.suite)
    rows = [_board_row(i + 1, r) for i, r in enumerate(service.leaderboard(ctx.ws, cfg))]
    return {
        "suite": cfg.eval.suite,
        "n_policies": len(rows),
        "rows": rows[: args.top] if args.top > 0 else rows,
    }


def _compare(ctx: ToolContext, args: CompareArgs) -> dict[str, Any]:
    cfg = _with_suite(ctx.cfg, args.suite)
    for name in (args.a, args.b):
        get_artifact(ctx.ws, name, kind="policy")
    board = {r.policy: _board_row(i + 1, r) for i, r in enumerate(service.leaderboard(ctx.ws, cfg))}
    for name in (args.a, args.b):
        if name not in board:
            raise NotFoundError(f"策略 {name} 在评测集 {cfg.eval.suite} 上还没有评测结果")
    # 配对：同样的步数上限与种子下的同一个初始状态。n_exec 不算进去：它随策略的分块长度而变
    suite = load_suite(ctx.ws, _task_name(ctx), cfg.eval.suite)
    outcomes: dict[str, dict[tuple[int, int, int], int]] = {args.a: {}, args.b: {}}
    for ev in eval_store.done_evals(ctx.ws, suite.id, service.REAL):
        if ev["policy"] in outcomes:
            for row in eval_store.episodes_of(ctx.ws, ev["id"]):
                key = (ev["max_steps"], ev["seed"], row["init_index"])
                outcomes[ev["policy"]][key] = row["success"]
    got_a, got_b = outcomes[args.a], outcomes[args.b]
    paired = sorted(got_a.keys() & got_b.keys())
    only_a = sum(1 for k in paired if got_a[k] and not got_b[k])
    only_b = sum(1 for k in paired if got_b[k] and not got_a[k])
    ra, rb = board[args.a], board[args.b]
    return {
        "suite": cfg.eval.suite,
        "a": ra,
        "b": rb,
        "ci_overlap": ra["ci95"][0] <= rb["ci95"][1] and rb["ci95"][0] <= ra["ci95"][1],
        "paired": {
            "n_pairs": len(paired),
            "only_a_succeeds": only_a,
            "only_b_succeeds": only_b,
            "mcnemar_p": round(mcnemar_exact(only_a, only_b), 6),
        },
    }


def _list_runs(ctx: ToolContext, args: ListRunsArgs) -> dict[str, Any]:
    limit = max(1, min(args.limit, MAX_LISTED_RUNS))
    rows = ctx.ws.db.execute(
        "SELECT id, kind, status, started_at, finished_at FROM runs WHERE (?1 = '' OR kind = ?1)"
        " ORDER BY started_at DESC, id DESC LIMIT ?2",
        (args.kind, limit),
    ).fetchall()
    return {"n_listed": len(rows), "runs": [dict(r) for r in rows]}


_META_KEYS = ("policy", "backend", "sim", "suite", "seed", "device", "git_rev", "git_dirty", "dataset_fp")


def _get_run(ctx: ToolContext, args: GetRunArgs) -> dict[str, Any]:
    row = ctx.ws.db.execute(
        "SELECT id, kind, status, started_at, finished_at FROM runs WHERE id = ?", (args.run_id,)
    ).fetchone()
    if row is None:
        raise NotFoundError(f"没有编号为 {args.run_id} 的运行")
    run_dir = ctx.ws.root / "runs" / row["id"]  # 用库里的编号拼目录，不用模型给的字符串
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    last: dict[str, dict[str, Any]] = {}
    for m in read_metrics(run_dir):
        last[m["name"]] = {"step": m["step"], "value": round(m["value"], 6)}
    return {
        **dict(row),
        "meta": {k: meta[k] for k in _META_KEYS if meta.get(k) is not None},
        "last_metrics": last,
    }


def _dataset_info(ctx: ToolContext, args: DatasetArgs) -> dict[str, Any]:
    name = args.name or ctx.cfg.data.name
    m = load_dataset(ctx.ws, _task_name(ctx), name).manifest
    return {
        "name": name,
        "n_episodes": m["n_episodes"],
        "n_frames": m["n_frames"],
        "fingerprint": m["fingerprint"][:12],
        "splits": {split: len(indices) for split, indices in m["splits"].items()},
    }


def _describe_config(ctx: ToolContext, args: DescribeConfigArgs) -> dict[str, Any]:
    if args.section.split(".")[0] == "llm":
        raise ToolError("llm 段（模型服务的接入方式）不在工具里展示")
    current = ctx.cfg.to_dict()
    items = []
    for item in describe_config(args.section):
        if item["key"].split(".")[0] in ("llm", "workspace"):
            continue
        value: Any = current
        for part in item["key"].split("."):
            value = value[part]
        items.append({**item, "current": value})
    return {"items": items}


def _check_overrides(ctx: ToolContext, args: CheckOverridesArgs) -> dict[str, Any]:
    checked_overrides(ctx.cfg, args.overrides)
    return {"ok": True, "overrides": [list(split_override(o)) for o in args.overrides]}


def _run_eval(ctx: ToolContext, args: RunEvalArgs) -> dict[str, Any]:
    out = service.run_eval(ctx.ws, _with_suite(ctx.cfg, args.suite), args.policy, force=args.force)
    return {
        "policy": out.policy,
        "suite": args.suite or ctx.cfg.eval.suite,
        "successes": out.successes,
        "n": out.n,
        "success_rate": round(out.successes / max(out.n, 1), 4),
        "reused_existing_result": out.skipped,
    }


def _train_policy(ctx: ToolContext, args: TrainPolicyArgs) -> dict[str, Any]:
    # 两种算法各读一段配置；改到另一段的覆盖项不会生效，所以直接拒绝，不让它悄悄被忽略
    section = TRAIN_SECTIONS[args.backend]
    cfg = checked_overrides(ctx.cfg, args.overrides, only=(section,))
    artifact = train_policy(ctx.ws, cfg, args.name, backend=args.backend)
    return {"name": artifact.name, "backend": artifact.backend, "run_id": artifact.run_id}


def _remove_policy(ctx: ToolContext, args: RemovePolicyArgs) -> dict[str, Any]:
    get_artifact(ctx.ws, args.name, kind="policy")
    return {"removed": args.name, "removed_evals": delete_artifact(ctx.ws, args.name)}


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "list_policies",
            "列出当前任务已登记的全部策略（名称、算法、登记时间）",
            NoArgs,
            "read",
            _list_policies,
        ),
        Tool(
            "leaderboard",
            "某个评测集上的排行榜：每个策略的成功数、成功率、95% 区间、平均最大进度",
            LeaderboardArgs,
            "read",
            _leaderboard,
        ),
        Tool(
            "compare_policies",
            "比较两个策略在同一个评测集上的结果：各自的成功率与区间、区间是否重叠、配对检验的 p 值",
            CompareArgs,
            "read",
            _compare,
        ),
        Tool(
            "list_runs", "列出最近的运行记录（编号、类型、状态、起止时间）", ListRunsArgs, "read", _list_runs
        ),
        Tool(
            "get_run", "查看一次运行的详情：状态、元信息、各项指标的最后一个值", GetRunArgs, "read", _get_run
        ),
        Tool(
            "dataset_info",
            "数据集的概况：回合数、帧数、指纹、训练/验证/测试各多少回合",
            DatasetArgs,
            "read",
            _dataset_info,
        ),
        Tool(
            "describe_config",
            "列出配置项：键、类型、默认值、当前值、说明",
            DescribeConfigArgs,
            "read",
            _describe_config,
        ),
        Tool(
            "check_overrides",
            "校验一组配置覆盖项是否合法（键是否存在、类型是否正确）。只检查，不执行任何操作",
            CheckOverridesArgs,
            "read",
            _check_overrides,
        ),
        Tool("run_eval", "在真实仿真器里评测一个已登记的策略，结果入库", RunEvalArgs, "write", _run_eval),
        Tool(
            "train_policy",
            "训练一个新策略并登记。耗时较长，执行前需要用户确认",
            TrainPolicyArgs,
            "confirm",
            _train_policy,
        ),
        Tool(
            "remove_policy",
            "从注册表删除一个策略和它的全部评测结果。不可恢复，执行前需要用户确认",
            RemovePolicyArgs,
            "confirm",
            _remove_policy,
        ),
    )
}


def tool_specs(tools: Mapping[str, Tool] = TOOLS) -> list[ToolSpec]:
    return [t.spec() for t in tools.values()]


def normalized_args(tool: Tool, raw: Any) -> dict[str, Any]:
    """校验并规范化参数：补默认值、去首尾空白、数值按类型。没通过校验时抛 ToolError。"""
    return asdict(parse_args(tool.args, raw))


def execute(
    ctx: ToolContext,
    name: str,
    raw_args: Any,
    *,
    allow: Collection[Access] = ("read",),
    tools: Mapping[str, Tool] = TOOLS,
) -> dict[str, Any]:
    """执行一次工具调用，返回给模型看的观察。出错不抛异常：那一行中文报错就是观察。

    allow 是这次调用得到许可的权限档。默认只许只读；可写与需确认的工具要调用方明确放行，
    模型自己说「用户已同意」不起作用。
    """
    try:
        if name not in tools:
            raise ToolError(f"没有名为 {name} 的工具；可用的工具有：{'、'.join(tools)}")
        tool = tools[name]
        args = parse_args(tool.args, raw_args)
        if tool.access not in allow:
            raise ToolError(
                f"工具 {name} 属于「{ACCESS_ZH[tool.access]}」一档，这次调用没有得到许可，没有执行"
            )
        return tool.run(ctx, args)
    except RoboArenaError as exc:
        return {"error": str(exc)}
