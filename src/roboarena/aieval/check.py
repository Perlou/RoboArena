"""检查器：判定模型的回答对不对。纯函数，先过自己的测试（正确的判过，故意做错的判不过）。

toolcall 的两个指标（定义见 docs/PLAN.md 1.2 节）：
    合法      有工具调用，且每一次调用的工具存在、参数通过类型校验
    完全匹配  合法，且调用的次数、工具名、规范化之后的全部参数都与标准答案一致
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from roboarena.agent.tools import TOOLS, Tool, normalized_args
from roboarena.aieval.cases import Case
from roboarena.config import split_override
from roboarena.core.errors import ConfigError, ToolError
from roboarena.core.hashing import canonical_json
from roboarena.llm.api import ToolCall

SYMMETRIC = {"compare_policies": ("a", "b")}  # 这两个参数互换，调用的意思不变


@dataclass(frozen=True)
class ToolcallScore:
    valid: bool
    match: bool
    reason: str  # 没通过时说明原因；通过时为空


def _number(value: Any) -> Any:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else value


def _canonical_override(item: str) -> str:
    """覆盖项按（键，值）比较：policy.lr=1e-3 与 policy.lr = 0.001 是同一项。"""
    try:
        key, value = split_override(item)
    except ConfigError:
        return item
    return canonical_json([key, _number(value)])


def canonical_call(call: ToolCall, tools: Mapping[str, Tool] = TOOLS) -> tuple[str, dict[str, Any]]:
    """规范化一次调用：补默认值、去首尾空白、数值按类型；覆盖项不计顺序；对称的参数排好序。

    工具不存在或参数没通过校验时抛 ToolError。
    """
    if call.name not in tools:
        raise ToolError(f"工具不存在：{call.name}")
    if call.raw_arguments is not None:
        raise ToolError("参数不是合法的 JSON 对象")
    args = normalized_args(tools[call.name], call.arguments)
    if isinstance(args.get("overrides"), list):
        args["overrides"] = sorted(_canonical_override(o) for o in args["overrides"])
    if call.name in SYMMETRIC:
        first, second = SYMMETRIC[call.name]
        args[first], args[second] = sorted((args[first], args[second]))
    return call.name, args


def check_toolcall(case: Case, calls: Sequence[ToolCall], tools: Mapping[str, Tool] = TOOLS) -> ToolcallScore:
    if case.gold_calls is None:
        raise ValueError(f"用例 {case.id} 没有标准工具调用，不能用 toolcall 检查器")
    if not calls:
        return ToolcallScore(False, False, "没有工具调用")
    try:
        got = [canonical_call(c, tools) for c in calls]
    except ToolError as exc:
        return ToolcallScore(False, False, f"调用不合法：{exc}")
    gold = [canonical_call(c, tools) for c in case.gold_calls]
    if len(got) != len(gold):
        return ToolcallScore(True, False, f"应调用 {len(gold)} 次工具，实际 {len(got)} 次")
    for (name, args), (gold_name, gold_args) in zip(got, gold, strict=True):
        if name != gold_name:
            return ToolcallScore(True, False, f"工具不对：应为 {gold_name}，实际 {name}")
        if wrong := [k for k in gold_args if args[k] != gold_args[k]]:
            detail = "；".join(f"{k} 应为 {gold_args[k]!r}，实际 {args[k]!r}" for k in wrong)
            return ToolcallScore(True, False, f"参数不一致：{detail}")
    return ToolcallScore(True, True, "")
