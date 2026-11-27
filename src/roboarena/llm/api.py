"""语言模型的统一接口（docs/ARCHITECTURE.md 16.2 节）。适配器实现 ChatModel，上层只认这几个类型。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # 模型给的参数不是一个合法的 JSON 对象时，原文放在这里，arguments 为空。这不算调用失败，由上层决定怎么办
    raw_arguments: str | None = None


@dataclass(frozen=True)
class Message:
    role: str  # system / user / assistant / tool
    content: str
    tool_calls: tuple[ToolCall, ...] = ()  # 只有 assistant 消息会带
    tool_call_id: str | None = None  # 只有 tool 消息会带：这是哪一次调用的结果


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...]
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool


class ChatModel(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Completion: ...
