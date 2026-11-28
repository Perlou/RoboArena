"""项目自己的提示渲染与输出解析：ChatML 对话格式加 <tool_call> 工具调用（Qwen 系列用的约定）。

走原样补全的模型都用这一套，所以它们看到的工具描述逐字相同，输出也按同一套规则解析（docs/PLAN.md 1.3 节）。
工具说明那一段的文字照 Qwen 官方对话模板；工具描述是不转义中文的 JSON，一行一个。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from roboarena.llm.api import Message, ToolCall, ToolSpec

_START, _END = "<|im_start|>", "<|im_end|>"
_TOOLS_HEAD = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
)
_TOOLS_TAIL = (
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
_THINK_PAIR = re.compile(r"<think>.*?</think>", re.DOTALL)
_CALL_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _tool_block(tools: Sequence[ToolSpec]) -> str:
    lines = [
        _dumps(
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
        )
        for t in tools
    ]
    return _TOOLS_HEAD + "\n".join(lines) + _TOOLS_TAIL


def _call_body(call: ToolCall) -> str:
    """一次调用在历史里的写法。模型上一轮写坏了的调用原样写回去，它才看得懂随后的报错。"""
    if call.raw_arguments is not None and not call.name:
        return call.raw_arguments  # 整块都没解析出来
    arguments = call.raw_arguments if call.raw_arguments is not None else _dumps(call.arguments)
    return f'{{"name": "{call.name}", "arguments": {arguments}}}'


def render(messages: Sequence[Message], tools: Sequence[ToolSpec] = (), *, prefix: str = "") -> str:
    """把对话渲染成交给模型续写的提示。prefix 接在助手开头之后（带思考的模型用 "<think>\\n"）。"""
    parts: list[str] = []
    rest = list(messages)
    system = rest.pop(0).content if rest and rest[0].role == "system" else None
    if tools:
        head = f"{system}\n\n" if system is not None else ""
        parts.append(f"{_START}system\n{head}{_tool_block(tools)}{_END}\n")
    elif system is not None:
        parts.append(f"{_START}system\n{system}{_END}\n")
    for i, message in enumerate(rest):
        if message.role == "tool":  # 连续的工具结果合在一条 user 消息里
            if i == 0 or rest[i - 1].role != "tool":
                parts.append(f"{_START}user")
            parts.append(f"\n<tool_response>\n{message.content}\n</tool_response>")
            if i == len(rest) - 1 or rest[i + 1].role != "tool":
                parts.append(f"{_END}\n")
        elif message.role == "assistant" and message.tool_calls:
            parts.append(f"{_START}assistant")
            if message.content:
                parts.append(f"\n{message.content}")
            parts += [f"\n<tool_call>\n{_call_body(c)}\n</tool_call>" for c in message.tool_calls]
            parts.append(f"{_END}\n")
        elif message.role in ("user", "assistant", "system"):
            parts.append(f"{_START}{message.role}\n{message.content}{_END}\n")
        else:
            raise ValueError(f"不认识的消息角色：{message.role}")
    parts.append(f"{_START}assistant\n{prefix}")
    return "".join(parts)


def _parse_block(index: int, body: str) -> ToolCall:
    call_id = f"call_{index}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"]:
        return ToolCall(call_id, "", {}, raw_arguments=body)
    arguments = data.get("arguments")
    if arguments is None:
        return ToolCall(call_id, data["name"], {})
    if not isinstance(arguments, dict):  # 参数必须是 JSON 对象；别的形状重新写成 JSON 文本交给上层
        return ToolCall(call_id, data["name"], {}, raw_arguments=_dumps(arguments))
    return ToolCall(call_id, data["name"], arguments)


def parse(output: str) -> tuple[str, tuple[ToolCall, ...]]:
    """把模型的输出拆成（文字，工具调用）。思考的内容丢掉，思考里提到的调用不算数。

    解析不了的 <tool_call> 块不丢：整块解析不了时工具名为空、原文在 raw_arguments 里；
    参数不是 JSON 对象时工具名照旧、参数的文本在 raw_arguments 里。没有标签的裸 JSON 不算调用。
    """
    text = _THINK_PAIR.sub("", output)
    if "</think>" in text:  # 开头的 <think> 写在提示里、调用方没拼回来
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:  # 思考没结束，输出就到了上限或被中止：后面的全是思考
        text = text.split("<think>", 1)[0]
    calls: list[ToolCall] = []
    pieces: list[str] = []
    position = 0
    for match in _CALL_BLOCK.finditer(text):
        pieces.append(text[position : match.start()])
        calls.append(_parse_block(len(calls), match.group(1).strip()))
        position = match.end()
    pieces.append(text[position:])
    kept = [p.strip() for p in pieces if p.strip()]
    return "\n".join(kept), tuple(calls)
