"""运行 AI 评测：同一组用例交给一个模型配置，逐用例打分，结果入库（docs/ARCHITECTURE.md 16.4 节）。

一次评测是一条 aieval 类型的运行记录加 ai_evals 的一行；每个用例的完整对话在运行目录的 cases.jsonl 里。
（用例集指纹，模型，配置，种子）相同的评测不重复记录。
"""

from __future__ import annotations

import copy
import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from roboarena.agent.loop import AGENT_PROMPT, Session, run_turn
from roboarena.agent.tools import ToolContext, tool_specs
from roboarena.aieval.cases import Case, suite_fingerprint
from roboarena.aieval.check import check_toolcall
from roboarena.aieval.fixture import copy_fixture, ensure_fixture, on_cpu
from roboarena.aieval.tasks import TaskCase, check_task, snapshot, task_fingerprint
from roboarena.config import Config
from roboarena.core.hashing import canonical_json, sha256_text
from roboarena.core.log import get_logger
from roboarena.llm.api import Message, ToolSpec
from roboarena.llm.gateway import Gateway, read_calls
from roboarena.store import aievals as store
from roboarena.store.runs import start_run
from roboarena.store.workspace import Workspace

log = get_logger(__name__)

TOOLCALL = "toolcall"
TASK = "task"
SUITES = (TOOLCALL, TASK)
MAX_TOKENS = 512
# 所有模型配置共用这一份系统提示、同一套工具描述（docs/PLAN.md 1.3 节）
SYSTEM_PROMPT = (
    "你是 RoboArena 机器人策略评测平台的助手。用户的请求要通过平台的工具完成。"
    "每次只调用一个最合适的工具，不要用文字代替工具调用。"
    "参数只填用户明确给出的内容；用户没有提到的可选参数不要填。"
    "名称（策略、评测集、数据集、运行编号）照用户的原文填写，不要改写。"
)


@dataclass(frozen=True)
class AiEvalOutcome:
    eval_id: int
    model: str
    seed: int
    n: int
    valid: int
    matched: int
    skipped: bool  # 库里已有同一次评测的结果，没有重跑


@dataclass(frozen=True)
class TaskEvalOutcome:
    eval_id: int
    model: str
    setup: str
    seed: int
    n: int
    passed: int
    answers: int
    hallucinated_answers: int
    violations: int
    skipped: bool  # 库里已有同一次评测的结果，没有重跑


def setup_id(
    name: str, system: str, tools: Sequence[ToolSpec], model_id: str, max_tokens: int = MAX_TOKENS
) -> str:
    """配置的标识：名称加上系统提示、工具描述、输出上限、模型身份的摘要。

    这些一变就是另一种配置，不会和旧结果混在一起。模型身份（后端、服务地址、模型标识、助手开头）算进来，
    是因为配置里同一个名称可以改指别的后端或别的服务：提示由谁渲染变了，结果就不能当成同一次评测。
    """
    digest = sha256_text(
        canonical_json(
            {
                "system": system,
                "tools": [asdict(t) for t in tools],
                "max_tokens": max_tokens,
                "model": model_id,
            }
        )
    )
    return f"{name}-{digest[:8]}"


def task_setup_id(
    name: str,
    system: str,
    tools: Sequence[ToolSpec],
    agent_cfg: Mapping[str, Any],
    model_id: str,
) -> str:
    """task 套件的配置标识：系统提示、工具描述、Agent 循环参数（含结果校验开关）、模型身份的摘要。"""
    digest = sha256_text(
        canonical_json(
            {
                "system": system,
                "tools": [asdict(t) for t in tools],
                "agent": dict(agent_cfg),
                "model": model_id,
            }
        )
    )
    return f"{name}-{digest[:8]}"


def run_toolcall(
    ws: Workspace,
    cfg: Config,
    gateway: Gateway,
    model: str,
    cases: Sequence[Case],
    *,
    seed: int = 0,
    setup: str = "zeroshot",
    force: bool = False,
) -> AiEvalOutcome:
    """在 toolcall 用例上评测一个模型。temperature 固定为 0，种子写进请求。

    gateway 不应配置备用模型：降级之后记下来的就不是这个模型的回答了。
    """
    if gateway.cfg.fallback:
        raise ValueError("评测用的网关不能配置备用模型")
    specs = tool_specs()
    backend = gateway.cfg.models[model].backend if model in gateway.cfg.models else "?"
    setup_name = setup_id(f"{setup}/{backend}", SYSTEM_PROMPT, specs, gateway.model_id(model))
    key = (suite_fingerprint(cases), model, setup_name, seed)
    if (existing := store.find_ai_eval(ws, key)) is not None:
        if existing["status"] == "done" and not force:
            old = [r for r in store.ai_case_rows(ws, TOOLCALL) if r["eval_id"] == existing["id"]]
            valid = sum(json.loads(r["scores_json"])["valid"] for r in old)
            matched = sum(r["passed"] for r in old)
            return AiEvalOutcome(existing["id"], model, seed, len(old), valid, matched, True)
        store.delete_ai_eval(ws, existing["id"])  # 上次没跑完，或要求重跑

    with start_run(ws, "aieval", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=seed) as run:
        run.update_meta(suite=TOOLCALL, suite_fp=key[0], model=model, setup=key[2], n_cases=len(cases))
        previous = gateway.calls_path
        gateway.calls_path = run.dir / "llm_calls.jsonl"
        eval_id = store.create_ai_eval(ws, TOOLCALL, key, run.id)
        rows: list[store.AiCaseRow] = []
        valid = matched = 0
        try:
            with open(run.dir / "cases.jsonl", "w", encoding="utf-8") as transcript:
                for case in cases:
                    messages = [Message("system", SYSTEM_PROMPT), Message("user", case.turns[0])]
                    done = gateway.complete(
                        messages, specs, model=model, temperature=0.0, max_tokens=MAX_TOKENS, seed=seed
                    )
                    score = check_toolcall(case, done.tool_calls)
                    valid += score.valid
                    matched += score.match
                    scores = {**asdict(score), "cached": done.cached, "finish": done.finish_reason}
                    rows.append((
                        case.id, case.family, case.bucket, score.match, canonical_json(scores),
                        done.latency_ms, done.tokens_in, done.tokens_out, 1,
                    ))  # fmt: skip
                    record = {
                        "case": asdict(case),
                        "messages": [asdict(m) for m in messages],
                        "completion": asdict(done),
                        "scores": scores,
                    }
                    transcript.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            gateway.calls_path = previous
        store.save_ai_cases(ws, eval_id, rows)
        run.log_metric(0, "valid_rate", valid / max(len(cases), 1))
        run.log_metric(0, "match_rate", matched / max(len(cases), 1))
        log.info(
            "AI 评测完成",
            extra={"model": model, "seed": seed, "n": len(cases), "valid": valid, "matched": matched},
        )
    return AiEvalOutcome(eval_id, model, seed, len(cases), valid, matched, False)


def run_task(
    ws: Workspace,
    cfg: Config,
    gateway: Gateway,
    model: str,
    cases: Sequence[TaskCase],
    *,
    seed: int = 0,
    setup: str = "",
    force: bool = False,
    fixture_root: Path | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> TaskEvalOutcome:
    """在 task 多步用例上评测 Agent。每个用例在固定工作区的一份独立副本上运行，互不影响。"""
    if gateway.cfg.fallback:
        raise ValueError("评测用的网关不能配置备用模型")
    specs = tool_specs()
    backend = gateway.cfg.models[model].backend if model in gateway.cfg.models else "?"
    label = setup or ("verify" if cfg.agent.verify else "no-verify")
    setup_name = task_setup_id(
        f"{label}/{backend}", AGENT_PROMPT, specs, asdict(cfg.agent), gateway.model_id(model)
    )
    key = (task_fingerprint(cases), model, setup_name, seed)
    if (existing := store.find_ai_eval(ws, key)) is not None:
        if existing["status"] == "done" and not force:
            old = [r for r in store.ai_case_rows(ws, TASK) if r["eval_id"] == existing["id"]]
            parsed = [json.loads(r["scores_json"]) for r in old]
            return TaskEvalOutcome(
                existing["id"],
                model,
                setup_name,
                seed,
                len(old),
                sum(r["passed"] for r in old),
                sum(s["answers"] for s in parsed),
                sum(s["hallucinated_answers"] for s in parsed),
                sum(s["violations"] for s in parsed),
                True,
            )
        store.delete_ai_eval(ws, existing["id"])

    base_fixture = fixture_root if fixture_root is not None else ensure_fixture(ws.root / "cache")
    with start_run(ws, "aieval", config=cfg.to_dict(), config_toml=cfg.to_toml(), seed=seed) as run:
        run.update_meta(
            suite=TASK,
            suite_fp=key[0],
            model=model,
            setup=setup_name,
            verify=cfg.agent.verify,
            n_cases=len(cases),
        )
        calls_file = run.dir / "llm_calls.jsonl"
        previous = gateway.calls_path
        gateway.calls_path = calls_file
        eval_id = store.create_ai_eval(ws, TASK, key, run.id)
        rows: list[store.AiCaseRow] = []
        passed = answers = hallucinated = numbers = unsupported = violations = 0
        try:
            with open(run.dir / "cases.jsonl", "w", encoding="utf-8") as transcript, on_cpu():
                for case in cases:
                    before_calls = len(read_calls(calls_file))
                    with tempfile.TemporaryDirectory() as tmp:
                        case_root = Path(tmp) / "ws"
                        case_cfg = copy_fixture(base_fixture, case_root)
                        case_cfg.agent = copy.deepcopy(cfg.agent)
                        confirm = {
                            "yes": lambda tool, args: True,
                            "no": lambda tool, args: False,
                            "none": None,
                        }[case.confirm]
                        t0 = clock()
                        with Workspace(case_root) as case_ws:
                            ctx = ToolContext(case_ws, case_cfg)
                            before = snapshot(ctx)
                            session = Session.new(AGENT_PROMPT)
                            turns = tuple(
                                run_turn(ctx, gateway, session, text, model=model, confirm=confirm, seed=seed)
                                for text in case.turns
                            )
                            score = check_task(case, ctx, before, turns)
                        elapsed_ms = (clock() - t0) * 1000.0
                    ok_calls = [c for c in read_calls(calls_file)[before_calls:] if c["ok"]]
                    any_cached = any(c["cached"] for c in ok_calls)
                    llm_ms = sum(t.latency_ms for t in turns)
                    case_ms = 0.0 if any_cached else round(max(elapsed_ms, llm_ms), 3)
                    tokens_in = sum(t.tokens_in for t in turns)
                    tokens_out = sum(t.tokens_out for t in turns)
                    steps = sum(t.steps for t in turns)
                    scores = {
                        **asdict(score),
                        "stopped": [t.stopped for t in turns],
                        "retried": sum(t.retried for t in turns),
                        "cached": any_cached,
                        "call_ms": [float(c["latency_ms"]) for c in ok_calls if not c["cached"]],
                        "cost": sum(float(c["cost"]) for c in ok_calls),
                        "n_length": sum(c.get("finish_reason") == "length" for c in ok_calls),
                        "n_aborted": sum(c.get("finish_reason") == "aborted" for c in ok_calls),
                    }
                    passed += score.passed
                    answers += score.answers
                    hallucinated += score.hallucinated_answers
                    numbers += score.numbers
                    unsupported += score.unsupported_numbers
                    violations += score.violations
                    rows.append((
                        case.id, case.family, "all", score.passed, canonical_json(scores),
                        case_ms, tokens_in, tokens_out, steps,
                    ))  # fmt: skip
                    record = {
                        "case": asdict(case),
                        "messages": [asdict(m) for m in session.messages],
                        "turns": [asdict(t) for t in turns],
                        "scores": scores,
                        "latency_ms": case_ms,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                    }
                    transcript.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        finally:
            gateway.calls_path = previous
        store.save_ai_cases(ws, eval_id, rows)
        run.log_metric(0, "pass_rate", passed / max(len(cases), 1))
        run.log_metric(0, "hallucination_answer_rate", hallucinated / max(answers, 1))
        run.log_metric(0, "hallucination_number_rate", unsupported / max(numbers, 1))
        run.log_metric(0, "violations", float(violations))
        log.info(
            "AI 任务评测完成",
            extra={
                "model": model,
                "setup": setup_name,
                "seed": seed,
                "n": len(cases),
                "passed": passed,
                "hallucinated_answers": hallucinated,
                "violations": violations,
            },
        )
    return TaskEvalOutcome(
        eval_id, model, setup_name, seed, len(cases), passed, answers, hallucinated, violations, False
    )
