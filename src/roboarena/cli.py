"""命令行入口：roboarena <命令> [子命令] [--config 文件] [--set 键=值 ...]"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import MISSING, fields
from typing import Any

from roboarena import __version__
from roboarena.agent.chat import open_chat
from roboarena.agent.loop import ToolEvent, Turn
from roboarena.agent.tools import ACCESS_ZH, TOOLS, Tool
from roboarena.aieval.cases import BUCKETS, toolcall_cases
from roboarena.aieval.report import render_report as render_ai_report
from roboarena.aieval.run import SUITES as AI_SUITES
from roboarena.aieval.run import TASK, TOOLCALL, run_task, run_toolcall
from roboarena.aieval.tasks import task_cases
from roboarena.config import Config, load_config, with_overrides
from roboarena.core.device import get_device, is_quick
from roboarena.core.errors import RoboArenaError
from roboarena.core.log import setup_logging
from roboarena.data.dataset import build_dataset, load_dataset, render_report, verify_dataset
from roboarena.eval import service
from roboarena.eval.rollouts import collect_rollouts
from roboarena.llm.api import Message
from roboarena.llm.gateway import Gateway
from roboarena.policies import list_policy_backends
from roboarena.policies.train import train_policy
from roboarena.policies.zoo import build_zoo
from roboarena.store.registry import delete_artifact, list_artifacts, register_artifact
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task, list_tasks

# 还没实现的命令 → 计划在哪个里程碑完成
PLANNED = {
    "report": "M4",
    "pipeline": "M4",
    "bench": "M5a",
    "serve": "M5a",
    "token": "M5a",
}


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join(lines)


def cmd_info(cfg: Config, args: argparse.Namespace) -> int:
    print(f"RoboArena {__version__}")
    print(f"设备：{get_device()}")
    print(f"冒烟模式：{'是' if is_quick() else '否'}")
    print(f"当前任务：{cfg.task.name}")
    print(f"已注册的任务：{'、'.join(list_tasks())}")
    with Workspace(cfg.workspace.root) as ws:
        print(f"工作区：{ws.root}")
        for table, label in (
            ("runs", "运行"),
            ("artifacts", "策略与模型"),
            ("suites", "评测集"),
            ("evals", "评测"),
        ):
            count = ws.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {label}：{count}")
    return 0


def cmd_task_list(cfg: Config, args: argparse.Namespace) -> int:
    for name in list_tasks():
        try:
            spec = get_task(name).spec
        except ImportError as e:
            print(f"{name}：无法加载（{e}）")
            continue
        print(f"{name}：{json.dumps(spec.to_dict(), ensure_ascii=False)}")
    return 0


def cmd_data_suite(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        suite = service.create_suite(ws, cfg)
    print(
        f"已生成评测集 {suite.task}/{suite.name}：{len(suite.inits)} 个初始状态，指纹 {suite.fingerprint[:12]}"
    )
    return 0


def cmd_data_build(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        dataset = build_dataset(ws, cfg, force=args.force)
    m = dataset.manifest
    print(
        f"已生成数据集 {m['task']}/{m['name']}：{m['n_episodes']} 个回合，{m['n_frames']} 帧，指纹 {m['fingerprint'][:12]}"
    )
    print(f"校验报告：{dataset.dir / 'REPORT.md'}")
    return 0


def cmd_data_info(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        dataset = load_dataset(ws, get_task(cfg.task.name).spec.name, cfg.data.name)
    print(render_report(dataset), end="")
    return 0


def cmd_data_verify(cfg: Config, args: argparse.Namespace) -> int:
    task = get_task(cfg.task.name)
    with Workspace(cfg.workspace.root) as ws:
        dataset = load_dataset(ws, task.spec.name, cfg.data.name)
        problems = verify_dataset(dataset, task)
    if problems:
        print(f"数据集 {task.spec.name}/{cfg.data.name} 有 {len(problems)} 个问题：", file=sys.stderr)
        print("\n".join(f"  {p}" for p in problems[:20]), file=sys.stderr)
        return 1
    print(
        f"数据集 {task.spec.name}/{cfg.data.name} 核对通过：{dataset.manifest['n_episodes']} 个回合，指纹 {dataset.fingerprint[:12]}"
    )
    return 0


def cmd_train_policy(cfg: Config, args: argparse.Namespace) -> int:
    if args.resume is None and args.name is None:
        raise RoboArenaError("需要 --name（新训练）或 --resume <运行编号>（续训）")
    with Workspace(cfg.workspace.root) as ws:
        artifact = train_policy(ws, cfg, args.name or "", backend=args.backend, resume=args.resume)
    print(
        f"已训练并登记策略 {artifact.name}：{artifact.params['n_demos']} 条演示，{artifact.params['steps']} 步"
    )
    return 0


def cmd_train_zoo(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        names, created = build_zoo(ws, cfg)
    print(f"策略库共 {len(names)} 个策略，这次新增 {len(created)} 个：{'、'.join(created) or '无'}")
    return 0


def cmd_data_rollouts(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        summary = collect_rollouts(ws, cfg)
    rows = [(name, n, k, frames) for name, (n, k, frames) in summary.per_policy.items()]
    print(_table(("策略", "回合", "成功", "帧"), rows))
    print(f"已保存策略回合集 {cfg.rollouts.name}：{summary.dir}")
    print(
        f"并入数据集：roboarena data build --set data.name=<新名称> --set 'data.rollouts=[\"{cfg.rollouts.name}\"]'"
    )
    return 0


def cmd_policy_remove(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        n_evals = delete_artifact(ws, args.name)
    print(f"已删除策略 {args.name} 和它的 {n_evals} 次评测结果")
    return 0


def cmd_policy_add(cfg: Config, args: argparse.Namespace) -> int:
    params = json.loads(args.params)
    if not isinstance(params, dict):
        raise RoboArenaError("--params 应为一个 JSON 对象")
    task = get_task(cfg.task.name).spec.name
    with Workspace(cfg.workspace.root) as ws:
        artifact = register_artifact(
            ws, task=task, name=args.name, kind="policy", backend=args.backend, params=params
        )
    print(f"已登记策略 {artifact.name}（任务 {artifact.task}，后端 {artifact.backend}）")
    return 0


def cmd_policy_list(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        items = list_artifacts(ws, kind="policy")
    rows = [(a.name, a.task, a.backend, a.sha256[:12], a.created_at) for a in items]
    print(_table(("名称", "任务", "后端", "哈希", "登记时间"), rows) if rows else "还没有登记任何策略")
    return 0


def cmd_eval_run(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        names = args.policy or []
        if args.all:
            names = [
                a.name for a in list_artifacts(ws, task=get_task(cfg.task.name).spec.name, kind="policy")
            ]
        if not names:
            raise RoboArenaError("需要 --policy <名称…> 或 --all")
        for name in names:
            out = service.run_eval(ws, cfg, name, args.sim, force=args.force)
            note = "（库里已有结果，未重跑）" if out.skipped else ""
            print(f"{out.policy} @ {out.sim}：{out.successes}/{out.n} 成功{note}")
    return 0


def cmd_eval_leaderboard(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        board = service.leaderboard(ws, cfg, args.sim)
    if not board:
        print("这个评测集上还没有评测结果")
        return 0
    rows = [
        (i + 1, r.policy, f"{r.successes}/{r.n}", f"{r.rate:.3f}", f"[{r.ci_low:.3f}, {r.ci_high:.3f}]",
         f"{r.mean_max_progress:.3f}")
        for i, r in enumerate(board)
    ]  # fmt: skip
    print(f"任务 {cfg.task.name} ｜ 评测集 {cfg.eval.suite} ｜ 仿真器 {args.sim}")
    print(_table(("名次", "策略", "成功", "成功率", "95% Wilson 区间", "平均最大进度"), rows))
    return 0


def cmd_runs_list(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        found = ws.db.execute(
            "SELECT id, kind, status, git_rev, git_dirty, started_at FROM runs ORDER BY started_at DESC LIMIT ?",
            (args.limit,),
        ).fetchall()
    rows = [
        (
            r["id"],
            r["kind"],
            r["status"],
            (r["git_rev"] or "无")[:8] + ("*" if r["git_dirty"] else ""),
            r["started_at"],
        )
        for r in found
    ]
    print(_table(("运行", "类型", "状态", "代码版本", "开始时间"), rows) if rows else "还没有运行记录")
    return 0


def make_gateway(cfg: Config, ws: Workspace) -> Gateway:
    """命令行用的模型网关。缓存放在工作区的 cache/llm/ 下。"""
    return Gateway(cfg.llm, cache_dir=ws.root / "cache" / "llm")


def cmd_llm_list(cfg: Config, args: argparse.Namespace) -> int:
    rows = [
        (name, m.backend, m.model, m.base_url, "是" if name == cfg.llm.default else "")
        for name, m in cfg.llm.models.items()
    ]
    print(_table(("名称", "后端", "模型", "服务地址", "默认"), rows))
    return 0


def cmd_llm_ping(cfg: Config, args: argparse.Namespace) -> int:
    # 不用缓存、不重试、不降级：要看的就是这个模型现在通不通
    cfg = with_overrides(cfg, ["llm.cache=false", "llm.retries=0", 'llm.fallback=""'])
    failed = 0
    with Workspace(cfg.workspace.root) as ws:
        gateway = make_gateway(cfg, ws)
        for name in args.model or list(cfg.llm.models):
            try:
                done = gateway.complete([Message("user", "只回复一个字：好")], model=name, max_tokens=16)
            except RoboArenaError as exc:
                failed += 1
                print(f"{name}：失败：{exc}", flush=True)
            else:
                print(
                    f"{name}：正常，{done.latency_ms:.0f} 毫秒（输入 {done.tokens_in} / 输出 {done.tokens_out} token）",
                    flush=True,
                )
    return 1 if failed else 0


def cmd_agent_tools(cfg: Config, args: argparse.Namespace) -> int:
    def params(tool: Tool) -> str:
        names = [
            f.name + ("*" if f.default is MISSING and f.default_factory is MISSING else "")
            for f in fields(tool.args)
        ]
        return "、".join(names) or "（无）"

    rows = [(t.name, ACCESS_ZH[t.access], params(t), t.description) for t in TOOLS.values()]
    print(_table(("工具", "权限", "参数", "说明"), rows))
    return 0


def _event_line(event: ToolEvent, *, asked: bool) -> str:
    """一次工具调用的一行说明。asked 表示这次对话里能当面问用户（交互方式）。"""
    if event.status == "ok":
        note = "成功" + (("（用户已确认）" if asked else "（已预先同意）") if event.approved else "")
    elif event.status == "declined":
        note = "用户没有同意，未执行" if asked else "没有得到确认，未执行"
    else:
        label = {
            "error": "报错",
            "invalid": "调用不成立",
            "denied": "没有权限，未执行",
            "skipped": "超出上限，未执行",
        }
        note = label[event.status] + (
            f"：{event.observation.get('error')}" if event.status in ("error", "invalid") else ""
        )
    return f"· {event.name or '（无名）'} {json.dumps(event.args, ensure_ascii=False)} → {note}"


def _print_turn(turn: Turn, *, asked: bool) -> None:
    for event in turn.events:
        print(_event_line(event, asked=asked))
    print(turn.answer, flush=True)


def _ask(tool: Tool, tool_args: dict[str, Any]) -> bool:
    reply = input(f"确认执行 {tool.name} {json.dumps(tool_args, ensure_ascii=False)}？[y/N] ")
    return reply.strip().lower() in ("y", "yes", "是")


def cmd_chat(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        gateway = make_gateway(cfg, ws)
        with open_chat(ws, cfg, gateway, resume=args.resume, model=args.model) as chat:
            if args.once is not None:
                # 单次提问没有办法当面问用户：需确认的操作只有 --yes 预先同意了才执行
                turn = chat.say(args.once, confirm=(lambda tool, a: True) if args.yes else None)
                _print_turn(turn, asked=False)
                print(
                    f"（对话 {chat.run.id} ｜ 模型调用 {turn.steps} 次 ｜ 输入 {turn.tokens_in} / 输出 {turn.tokens_out} token）"
                )
                return 0
            print(f"（对话 {chat.run.id}，模型 {chat.model}；输入 exit 结束）")
            while True:
                try:
                    text = input("你> ").strip()
                except EOFError:
                    break
                if text in ("exit", "quit", "退出"):
                    break
                if not text:
                    continue
                try:
                    turn = chat.say(text, confirm=(lambda tool, a: True) if args.yes else _ask)
                except RoboArenaError as exc:  # 这一轮没成，对话还在，可以接着说
                    print(f"错误：{exc}", file=sys.stderr)
                    continue
                _print_turn(turn, asked=not args.yes)
            print(
                f"（对话 {chat.run.id} ｜ 共 {chat.turns} 轮；接着聊：roboarena chat --resume {chat.run.id}）"
            )
    return 0


def cmd_aieval_run(cfg: Config, args: argparse.Namespace) -> int:
    models = args.model or [cfg.llm.default]
    cfg = with_overrides(cfg, ['llm.fallback=""'])  # 评测不降级：降级之后记下来的就不是这个模型的回答了
    if args.suite == TASK:
        if args.bucket:
            raise RoboArenaError("task 套件不分桶，不能指定 --bucket")
        tcases = task_cases()
        if args.limit:
            tcases = tcases[:: max(1, len(tcases) // args.limit)][: args.limit]
        with Workspace(cfg.workspace.root) as ws:
            gateway = make_gateway(cfg, ws)
            for model in models:
                for seed in args.seeds:
                    out = run_task(ws, cfg, gateway, model, tcases, seed=seed, force=args.force)
                    note = "（库里已有结果，未重跑）" if out.skipped else ""
                    print(
                        f"{out.model}（{out.setup}）种子 {out.seed}：完成 {out.passed}/{out.n}，"
                        f"含幻觉回答 {out.hallucinated_answers}/{out.answers}，越权 {out.violations}{note}",
                        flush=True,
                    )
        return 0

    cases = toolcall_cases()
    if args.bucket:
        cases = [c for c in cases if c.bucket in args.bucket]
    if args.limit:
        cases = cases[:: max(1, len(cases) // args.limit)][: args.limit]  # 等间隔抽，各个族都抽到
    with Workspace(cfg.workspace.root) as ws:
        gateway = make_gateway(cfg, ws)
        for model in models:
            for seed in args.seeds:
                out = run_toolcall(ws, cfg, gateway, model, cases, seed=seed, force=args.force)
                note = "（库里已有结果，未重跑）" if out.skipped else ""
                print(
                    f"{out.model} 种子 {out.seed}：合法 {out.valid}/{out.n}，完全匹配 {out.matched}/{out.n}{note}",
                    flush=True,
                )
    return 0


def cmd_aieval_report(cfg: Config, args: argparse.Namespace) -> int:
    with Workspace(cfg.workspace.root) as ws:
        print(render_ai_report(ws, args.suite), end="")
    return 0


def cmd_planned(cfg: Config, args: argparse.Namespace) -> int:
    print(f"命令 {args.command} 尚未实现，计划在 {PLANNED[args.command]} 完成。", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="TOML 配置文件；不给时用默认值（冒烟模式下用 configs/quick.toml）")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="键=值",
                        help="覆盖单个配置项，可重复，如 --set eval.max_steps=50")  # fmt: skip

    parser = argparse.ArgumentParser(prog="roboarena", description="机器人策略评测平台")
    parser.add_argument("--version", action="version", version=f"roboarena {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="命令")

    def leaf(
        parent: argparse._SubParsersAction, name: str, help_: str, func: Callable
    ) -> argparse.ArgumentParser:
        p = parent.add_parser(name, help=help_, parents=[common])
        p.set_defaults(func=func)
        return p

    def group(name: str, help_: str) -> argparse._SubParsersAction:
        g = sub.add_parser(name, help=help_)
        return g.add_subparsers(dest="subcommand", required=True, metavar="子命令")

    leaf(sub, "info", "设备、工作区、已注册的任务", cmd_info)

    leaf(group("task", "任务插件"), "list", "列出已注册的任务及其规格", cmd_task_list)

    data = group("data", "数据与评测集")
    p = leaf(data, "build", "生成数据集：演示 + 仿真器内采集，校验、切分、算指纹", cmd_data_build)
    p.add_argument("--force", action="store_true", help="数据集已存在时删掉重建")
    leaf(data, "info", "显示数据集的校验报告", cmd_data_info)
    leaf(data, "verify", "重新读取全部回合文件，核对指纹并重跑检查", cmd_data_verify)
    leaf(data, "rollouts", "让已登记的策略在真实仿真器里跑，记录成可并入数据集的回合", cmd_data_rollouts)
    leaf(data, "suite", "按配置生成评测集（一组固定的初始状态）", cmd_data_suite)

    train = group("train", "训练")
    p = leaf(train, "policy", "训练一个策略并登记", cmd_train_policy)
    p.add_argument(
        "--backend",
        default="bc",
        choices=["bc", "diffusion"],
        help="bc 读 [policy] 段，diffusion 读 [diffusion] 段",
    )
    p.add_argument("--name", help="登记的名称")
    p.add_argument("--resume", metavar="运行编号", help="从这次运行的最后一个检查点接着训练")

    leaf(train, "zoo", "训练并登记整个策略库（已有的跳过）", cmd_train_zoo)

    policy = group("policy", "策略注册表")
    p = leaf(policy, "add", "登记一个不需要训练的策略", cmd_policy_add)
    p.add_argument("--name", required=True)
    p.add_argument("--backend", required=True, choices=list_policy_backends())
    p.add_argument("--params", default="{}", help="构造参数，JSON 对象，如 '{\"chunk\": 4}'")
    leaf(policy, "list", "列出已登记的策略", cmd_policy_list)
    p = leaf(policy, "remove", "删除一个策略和它的评测结果", cmd_policy_remove)
    p.add_argument("--name", required=True)

    ev = group("eval", "评测")
    p = leaf(ev, "run", "在指定仿真器上评测一个或多个策略", cmd_eval_run)
    p.add_argument("--policy", nargs="+", metavar="名称")
    p.add_argument("--all", action="store_true", help="评测当前任务的全部策略")
    p.add_argument("--sim", default=service.REAL, help="real，或某个世界模型的名称（M3 起）")
    p.add_argument("--force", action="store_true", help="已有结果时删掉重跑")
    p = leaf(ev, "leaderboard", "排行榜：成功率与 Wilson 区间", cmd_eval_leaderboard)
    p.add_argument("--sim", default=service.REAL)

    llm = group("llm", "模型网关")
    leaf(llm, "list", "配置里的语言模型", cmd_llm_list)
    p = leaf(llm, "ping", "给每个模型发一条最短的请求，看通不通", cmd_llm_ping)
    p.add_argument("--model", nargs="+", metavar="名称", help="只试这几个；省略表示全部")

    leaf(group("agent", "Agent"), "tools", "Agent 可用的工具、权限与参数（带 * 的必填）", cmd_agent_tools)

    p = leaf(sub, "chat", "用自然语言操作平台；不带 --once 时进入交互", cmd_chat)
    p.add_argument("--once", metavar="文本", help="只问这一句，答完就结束")
    p.add_argument("--resume", metavar="运行编号", help="接着这次对话继续")
    p.add_argument("--model", metavar="名称", help="配置里 [llm.models] 下的名称；省略时用 llm.default")
    p.add_argument("--yes", action="store_true", help="预先同意需确认的操作（训练、删除）")

    ai = group("aieval", "AI 评测：语言模型会不会操作这个平台")
    p = leaf(ai, "run", "在单步或多步用例上评测一个或多个模型，结果入库", cmd_aieval_run)
    p.add_argument(
        "--suite", default=TOOLCALL, choices=AI_SUITES, help="toolcall 单步工具调用，task 多步任务"
    )
    p.add_argument(
        "--model", nargs="+", metavar="名称", help="配置里 [llm.models] 下的名称；省略时用 llm.default"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], metavar="种子")
    p.add_argument("--bucket", nargs="+", choices=BUCKETS, help="只跑这些桶（仅 toolcall）")
    p.add_argument(
        "--limit", type=int, default=0, help="只跑这么多条（等间隔抽），试跑用；用例集的指纹会不同"
    )
    p.add_argument("--force", action="store_true", help="已有结果时删掉重跑")
    p = leaf(ai, "report", "由结果库生成对比报告（Markdown）", cmd_aieval_report)
    p.add_argument("--suite", default=TOOLCALL, choices=AI_SUITES)

    p = leaf(group("runs", "运行记录"), "list", "最近的运行", cmd_runs_list)
    p.add_argument("--limit", type=int, default=20)

    for name, milestone in PLANNED.items():
        p = leaf(sub, name, f"（尚未实现，{milestone}）", cmd_planned)
        p.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    try:
        cfg = load_config(args.config, args.overrides)
        return args.func(cfg, args)
    except RoboArenaError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
