"""运行 AI 评测：同一组用例交给一个模型配置，逐用例打分，结果入库（docs/ARCHITECTURE.md 16.4 节）。

一次评测是一条 aieval 类型的运行记录加 ai_evals 的一行；每个用例的完整对话在运行目录的 cases.jsonl 里。
（用例集指纹，模型，配置，种子）相同的评测不重复记录。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from roboarena.agent.tools import tool_specs
from roboarena.aieval.cases import Case, suite_fingerprint
from roboarena.aieval.check import check_toolcall
from roboarena.config import Config
from roboarena.core.hashing import canonical_json, sha256_text
from roboarena.core.log import get_logger
from roboarena.llm.api import Message, ToolSpec
from roboarena.llm.gateway import Gateway
from roboarena.store import aievals as store
from roboarena.store.runs import start_run
from roboarena.store.workspace import Workspace

log = get_logger(__name__)

TOOLCALL = "toolcall"
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
