"""评测的编排：查注册表与评测集 → 构造策略与仿真器 → 跑评测 → 结果入库。命令行和服务都调用这里。"""

from __future__ import annotations

from dataclasses import dataclass

from roboarena.config import Config
from roboarena.core.errors import ConflictError, RoboArenaError
from roboarena.core.log import get_logger
from roboarena.core.seeding import derive_seed, make_rng
from roboarena.eval.runner import evaluate_in_batches
from roboarena.eval.stats import wilson_interval
from roboarena.policies import build_policy
from roboarena.store import evals as eval_store
from roboarena.store.registry import checkpoint_path, get_artifact
from roboarena.store.runs import start_run
from roboarena.store.suites import Suite, has_suite, load_suite, save_suite
from roboarena.store.workspace import Workspace
from roboarena.tasks import get_task

log = get_logger(__name__)

REAL = "real"


@dataclass(frozen=True)
class EvalOutcome:
    eval_id: int
    policy: str
    sim: str
    n: int
    successes: int
    skipped: bool  # 库里已有同一次评测的结果，没有重跑


@dataclass(frozen=True)
class LeaderboardRow:
    policy: str
    n: int
    successes: int
    rate: float
    ci_low: float
    ci_high: float
    mean_max_progress: float


def create_suite(ws: Workspace, cfg: Config) -> Suite:
    """按配置生成评测集。初始状态的种子由（全局种子，评测集名称）派生，与数据采集用的种子不相交。"""
    task = get_task(cfg.task.name)
    if has_suite(ws, task.spec.name, cfg.eval.suite):
        raise ConflictError(f"任务 {task.spec.name} 已有名为 {cfg.eval.suite} 的评测集")
    inits = task.sample_inits(cfg.eval.n_inits, derive_seed(cfg.seed, f"suite:{cfg.eval.suite}"))
    return save_suite(ws, task.spec.name, cfg.eval.suite, [task.encode_init(i) for i in inits])


def run_eval(
    ws: Workspace, cfg: Config, policy_name: str, sim: str = REAL, *, force: bool = False
) -> EvalOutcome:
    """评测一个已注册的策略。同一次评测已有结果时直接返回，force=True 则删掉重跑。"""
    task = get_task(cfg.task.name)
    spec = task.spec
    artifact = get_artifact(ws, policy_name, kind="policy")
    if artifact.task != spec.name:
        raise RoboArenaError(f"策略 {policy_name} 属于任务 {artifact.task}，当前任务是 {spec.name}")
    if sim != REAL:
        raise RoboArenaError(f"暂时只支持真实仿真器（sim = {REAL}）；神经仿真器在 M3 加入")
    suite = load_suite(ws, spec.name, cfg.eval.suite)
    max_steps = cfg.eval.max_steps or spec.max_steps
    policy = build_policy(
        artifact.backend, spec, artifact.params, checkpoint_path(ws, artifact), artifact.name
    )
    n_exec = min(cfg.eval.n_exec, policy.chunk)  # 分块比 n_exec 短的策略，整块执行
    key = (artifact.id, sim, suite.id, max_steps, n_exec, cfg.eval.seed)

    if (existing := eval_store.find_eval(ws, key)) is not None:
        if existing["status"] == "done" and not force:
            rows = eval_store.episodes_of(ws, existing["id"])
            return EvalOutcome(
                existing["id"], policy_name, sim, len(rows), sum(r["success"] for r in rows), True
            )
        eval_store.delete_eval(ws, existing["id"])  # 上次没跑完，或要求重跑

    inits = [task.decode_init(d) for d in suite.inits]

    with start_run(ws, "eval", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=cfg.eval.seed) as run:
        run.update_meta(policy=policy_name, sim=sim, suite=suite.name, suite_fp=suite.fingerprint)
        eval_id = eval_store.create_eval(ws, key, run.id)
        results = evaluate_in_batches(
            policy,
            task.make_real_sim,
            inits,
            max_steps=max_steps,
            n_exec=n_exec,
            batch=cfg.eval.batch,
            make_rng=lambda start: make_rng(cfg.eval.seed, f"eval:{policy_name}:{start}"),
        )
        eval_store.save_episodes(
            ws, eval_id, [(i, r.success, r.max_progress, r.steps) for i, r in enumerate(results)]
        )
        successes = sum(r.success for r in results)
        run.log_metric(0, "success_rate", successes / max(len(results), 1))
        log.info(
            "评测完成", extra={"policy": policy_name, "sim": sim, "n": len(results), "successes": successes}
        )
    return EvalOutcome(eval_id, policy_name, sim, len(results), successes, False)


def leaderboard(ws: Workspace, cfg: Config, sim: str = REAL) -> list[LeaderboardRow]:
    """由逐回合记录现算。同一策略有多次评测（不同种子等）时合并统计。按成功率从高到低排。"""
    suite = load_suite(ws, get_task(cfg.task.name).spec.name, cfg.eval.suite)
    merged: dict[str, list[tuple[int, float]]] = {}
    for ev in eval_store.done_evals(ws, suite.id, sim):
        rows = eval_store.episodes_of(ws, ev["id"])
        merged.setdefault(ev["policy"], []).extend((r["success"], r["max_progress"]) for r in rows)
    board = []
    for policy, eps in merged.items():
        n, k = len(eps), sum(s for s, _ in eps)
        low, high = wilson_interval(k, n)
        board.append(LeaderboardRow(policy, n, k, k / n, low, high, sum(p for _, p in eps) / n))
    return sorted(board, key=lambda r: (-r.rate, -r.mean_max_progress, r.policy))
