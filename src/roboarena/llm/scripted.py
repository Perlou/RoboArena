"""脚本化假模型：按预先写好的脚本作答。测试用，也用来给 AI 评测算「预先知道的得分」。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from roboarena.core.errors import ModelError
from roboarena.llm.api import Completion, Message, ToolCall, ToolSpec

# 一条回答：一段文字、一次或几次工具调用，或者一个现成的 Completion（原样返回）
Reply = str | ToolCall | Sequence[ToolCall] | Completion
Responder = Callable[[Sequence[Message], Sequence[ToolSpec]], Reply]


@dataclass(frozen=True)
class Request:
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    temperature: float
    max_tokens: int
    seed: int


class ScriptedModel:
    """script 是回答的列表（按顺序各用一次），或一个由（消息，工具）给出回答的函数。

    token 数按字符数算：没有分词器，只求确定。收到的请求记在 requests 里，供测试核对。
    """

    def __init__(self, name: str, script: Sequence[Reply] | Responder) -> None:
        self.name = name
        self.requests: list[Request] = []
        self._script = script

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Completion:
        index = len(self.requests)
        self.requests.append(Request(tuple(messages), tuple(tools), temperature, max_tokens, seed))
        if callable(self._script):
            reply = self._script(messages, tools)
        elif index < len(self._script):
            reply = self._script[index]
        else:
            raise ModelError(
                f"假模型 {self.name} 的脚本只有 {len(self._script)} 条回答，第 {index + 1} 次调用没有可用的回答"
            )
        if isinstance(reply, Completion):
            return reply
        text = reply if isinstance(reply, str) else ""
        calls = () if isinstance(reply, str) else (reply,) if isinstance(reply, ToolCall) else tuple(reply)
        return Completion(
            text=text,
            tool_calls=calls,
            tokens_in=sum(len(m.content) for m in messages),
            tokens_out=len(text),
            latency_ms=0.0,
            cached=False,
        )
