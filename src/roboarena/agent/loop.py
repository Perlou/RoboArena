"""Agent 的循环（docs/ARCHITECTURE.md 16.3 节）：模型 → 工具调用 → 校验参数 → 查权限 → 执行 → 观察，直到给出回答。

三件事由代码把关，不靠提示词：
- 权限：只读角色只能用只读工具；需确认的工具在执行前问用户，问不到或用户不同意就不执行。
- 上限：模型调用次数、工具调用次数、token、时间，超出任何一项就停下，如实说明停在哪。
- 结果校验：回答里的数字找不到出处时退回重答一次，仍然找不到就照发并标明。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from roboarena.agent.tools import ACCESS_ZH, TOOLS, Access, Tool, ToolContext, execute, parse_args, tool_specs
from roboarena.agent.verify import Verdict, verify
from roboarena.core.errors import ToolError
from roboarena.llm.api import Message, ToolCall
from roboarena.llm.gateway import Gateway

AGENT_PROMPT = (
    "你是 RoboArena 机器人策略评测平台的助手，通过平台提供的工具帮用户查询与操作。\n"
    "- 需要平台里的事实（策略、成功率、运行记录、数据集、配置）时先调用工具，不要凭记忆或猜测回答。\n"
    "- 回答里的每个数字都必须出自本次对话的工具输出或用户的话，照抄即可，不要自己估算或换算。\n"
    "- 工具的参数只填用户明确给出的内容；名称（策略、评测集、数据集、运行编号）照用户的原文填写。\n"
    "- 训练与删除是需要确认的操作：直接发起工具调用即可，平台会去问用户。"
    "用户没有同意时如实告知，不要重试，也不要说已经执行。\n"
    "- 工具报错时，按报错改正参数，或如实告诉用户，不要编造结果。\n"
    "- 用中文简洁作答。"
)

Role = Literal["read", "write"]
ROLE_ACCESS: dict[str, tuple[Access, ...]] = {"read": ("read",), "write": ("read", "write")}
# ok 执行成功；error 执行了但报错；invalid 调用本身不成立；denied 角色不许；declined 需确认而没有得到同意；skipped 超出上限没有执行
Status = Literal["ok", "error", "invalid", "denied", "declined", "skipped"]
Confirm = Callable[[Tool, dict[str, Any]], bool]  # （工具，校验后的参数）→ 用户是否同意


@dataclass(frozen=True)
class ToolEvent:
    call_id: str
    name: str
    args: dict[str, Any]  # 校验后的参数；调用不成立时是模型给的原样
    access: Access | None  # 工具不存在时为 None
    status: Status
    approved: bool  # 这次执行是不是用户确认过的
    observation: dict[str, Any]


@dataclass(frozen=True)
class Turn:
    answer: str
    events: tuple[ToolEvent, ...]
    steps: int  # 调用了几次模型
    tokens_in: int
    tokens_out: int
    latency_ms: float
    stopped: (
        str  # answer 正常给出回答；max_steps / max_tool_calls / max_tokens / max_seconds 是撞到了哪个上限
    )
    retried: bool  # 结果校验有没有退回过一次
    verdict: Verdict  # 对最终回答的校验结果；校验关着也照样算


@dataclass
class Session:
    """一段对话。messages 是完整的记录；给模型看的版本由 _view 现做。"""

    messages: list[Message]
    sources: list[str] = field(default_factory=list)  # 数字的出处：用户说的话、工具输出的原文
    summaries: dict[int, str] = field(default_factory=dict)  # 工具消息的下标 → 超预算时代替它的一行摘要
    compact_before: int = 0  # 下标小于它的工具输出，给模型看的是摘要
    last_tokens_in: int = 0  # 上一次请求的输入 token 数

    @classmethod
    def new(cls, system: str = AGENT_PROMPT) -> Session:
        return cls([Message("system", system)])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        messages = [
            Message(m["role"], m["content"], tuple(ToolCall(**c) for c in m["tool_calls"]), m["tool_call_id"])
            for m in data["messages"]
        ]
        summaries = {int(index): text for index, text in data["summaries"].items()}  # JSON 的键只能是字符串
        return cls(messages, list(data["sources"]), summaries, data["compact_before"], data["last_tokens_in"])


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _summary(call: ToolCall, observation: dict[str, Any]) -> str:
    if "error" in observation:
        outcome = f"报错：{str(observation['error'])[:80]}"
    else:
        sizes = [len(v) for v in observation.values() if isinstance(v, list)]
        outcome = f"返回 {sizes[0]} 条" if sizes else "已返回"
    return f"（较早的工具输出已省略：{call.name} {_dumps(call.arguments)} {outcome}；需要时请重新调用）"


def _view(session: Session) -> list[Message]:
    """给模型看的消息：较早的工具输出换成摘要，其余原样。"""
    return [
        Message("tool", session.summaries[i], tool_call_id=m.tool_call_id)
        if m.role == "tool" and i < session.compact_before and i in session.summaries
        else m
        for i, m in enumerate(session.messages)
    ]


def _refused(call: ToolCall, status: Status, message: str, tool: Tool | None = None) -> ToolEvent:
    return ToolEvent(
        call.id, call.name, call.arguments, tool.access if tool else None, status, False, {"error": message}
    )


def _run_call(ctx: ToolContext, call: ToolCall, role: str, confirm: Confirm | None) -> ToolEvent:
    if call.raw_arguments is not None:
        if not call.name:
            return _refused(call, "invalid", f"工具调用不是合法的 JSON：{call.raw_arguments[:200]}")
        return _refused(call, "invalid", f"工具调用的参数不是一个 JSON 对象：{call.raw_arguments[:200]}")
    if call.name not in TOOLS:
        return _refused(call, "invalid", f"没有名为 {call.name} 的工具；可用的工具有：{'、'.join(TOOLS)}")
    tool = TOOLS[call.name]
    try:
        args = asdict(parse_args(tool.args, call.arguments))
    except ToolError as exc:
        return _refused(call, "invalid", str(exc), tool)
    allow: tuple[Access, ...] = ROLE_ACCESS[role]
    approved = False
    if tool.access == "confirm":
        if "write" not in allow:
            return _refused(
                call, "denied", f"工具 {tool.name} 属于「需确认」一档，这次调用没有得到许可，没有执行", tool
            )
        if confirm is None:
            return _refused(
                call, "declined", f"工具 {tool.name} 需要用户确认，现在没有办法询问用户，没有执行", tool
            )
        if not confirm(tool, args):
            return _refused(call, "declined", f"用户没有同意执行 {tool.name}，没有执行", tool)
        allow = (*allow, "confirm")
        approved = True
    elif tool.access not in allow:
        message = f"工具 {tool.name} 属于「{ACCESS_ZH[tool.access]}」一档，这次调用没有得到许可，没有执行"
        return _refused(call, "denied", message, tool)
    observation = execute(ctx, tool.name, call.arguments, allow=allow)  # 再过一遍 execute 的把关
    return ToolEvent(
        call.id,
        tool.name,
        args,
        tool.access,
        "error" if "error" in observation else "ok",
        approved,
        observation,
    )


def _shown(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…（输出太长，后面的 {len(text) - limit} 个字符已省略；请缩小查询范围）"


def _stop_answer(reason: str, events: list[ToolEvent]) -> str:
    ran = Counter(e.name for e in events if e.status in ("ok", "error"))
    done = "、".join(f"{name} ×{n}" for name, n in ran.items())
    return f"没有完成：{reason}。" + (f"已经执行的工具调用：{done}。" if done else "还没有执行任何工具调用。")


def run_turn(
    ctx: ToolContext,
    gateway: Gateway,
    session: Session,
    user_text: str,
    *,
    model: str | None = None,
    role: Role = "write",
    confirm: Confirm | None = None,
    seed: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> Turn:
    """处理用户的一句话，直到模型给出不带工具调用的回答，或撞到某个上限。会往 session 里追加消息。

    confirm 为 None 表示没有办法询问用户：需确认的工具一律不执行。
    """
    limits = ctx.cfg.agent
    specs = tool_specs()
    session.messages.append(Message("user", user_text))
    session.sources.append(user_text)
    events: list[ToolEvent] = []
    steps = tokens_in = tokens_out = 0
    latency_ms = 0.0
    retried = False
    start = clock()

    def stop(kind: str, reason: str) -> Turn:
        answer = _stop_answer(reason, events)
        session.messages.append(Message("assistant", answer))
        return Turn(
            answer, tuple(events), steps, tokens_in, tokens_out, latency_ms, kind, retried, Verdict((), ())
        )

    while True:
        if steps >= limits.max_steps:
            return stop("max_steps", f"已达到模型调用次数的上限（{limits.max_steps} 次）")
        if tokens_in + tokens_out >= limits.max_tokens:
            return stop("max_tokens", f"已达到 token 用量的上限（{limits.max_tokens}）")
        if clock() - start >= limits.max_seconds:
            return stop("max_seconds", f"已达到时间上限（{limits.max_seconds:g} 秒）")
        if session.last_tokens_in > limits.context_tokens:  # 超预算：最近一轮之前的工具输出换成摘要
            recent = [i for i, m in enumerate(session.messages) if m.role == "assistant" and m.tool_calls]
            session.compact_before = max(session.compact_before, recent[-1] if recent else 0)
        done = gateway.complete(
            _view(session), specs, model=model, temperature=0.0, max_tokens=limits.answer_tokens, seed=seed
        )
        steps += 1
        tokens_in += done.tokens_in
        tokens_out += done.tokens_out
        latency_ms += done.latency_ms
        session.last_tokens_in = done.tokens_in

        if done.tool_calls:
            session.messages.append(Message("assistant", done.text, tool_calls=done.tool_calls))
            over = False
            for call in done.tool_calls:
                executed = sum(e.status != "skipped" for e in events)
                if executed >= limits.max_tool_calls:
                    over = True
                    event = _refused(
                        call, "skipped", f"已达到工具调用次数的上限（{limits.max_tool_calls} 次），没有执行"
                    )
                else:
                    event = _run_call(ctx, call, role, confirm)
                events.append(event)
                full = _dumps(event.observation)
                session.sources.append(full)
                session.summaries[len(session.messages)] = _summary(call, event.observation)
                session.messages.append(
                    Message("tool", _shown(full, limits.observation_chars), tool_call_id=call.id)
                )
            if over:
                return stop("max_tool_calls", f"已达到工具调用次数的上限（{limits.max_tool_calls} 次）")
            continue

        text = done.text or f"（模型没有给出回答；结束原因：{done.finish_reason}）"
        verdict = verify(text, session.sources)
        if limits.verify and not verdict.ok:
            listed = "、".join(dict.fromkeys(m.text for m in verdict.unsupported))
            if not retried:
                retried = True
                session.messages.append(Message("assistant", text))
                session.messages.append(
                    Message(
                        "user",
                        f"（系统校验）你的回答里有数字找不到出处：{listed}。"
                        "回答里的数字必须出自本次对话的工具输出或用户的话。"
                        "请重新回答：只用有出处的数字，需要时可以再调用工具。",
                    )
                )
                continue
            text = f"{text}\n\n（注意：以下数字没有在工具输出或你的输入里找到出处：{listed}）"
        session.messages.append(Message("assistant", text))
        return Turn(text, tuple(events), steps, tokens_in, tokens_out, latency_ms, "answer", retried, verdict)
