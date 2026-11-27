"""项目自己的提示渲染与输出解析（ChatML + <tool_call>）。走原样补全的模型都用这一套，所以它们看到的工具描述逐字相同。"""

import pytest

from roboarena.llm import chatml
from roboarena.llm.api import Message, ToolCall, ToolSpec

BOARD = ToolSpec(
    "leaderboard",
    "查看排行榜",
    {"type": "object", "properties": {"suite": {"type": "string", "description": "评测集"}}, "required": []},
)
POLICIES = ToolSpec("list_policies", "列出策略", {"type": "object", "properties": {}})

# 工具说明这一段照 Qwen 官方对话模板的文字；工具描述是不转义中文的 JSON，一行一个
TOOL_BLOCK = (
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    '{"type": "function", "function": {"name": "leaderboard", "description": "查看排行榜", "parameters": '
    '{"type": "object", "properties": {"suite": {"type": "string", "description": "评测集"}}, "required": []}}}\n'
    '{"type": "function", "function": {"name": "list_policies", "description": "列出策略", "parameters": '
    '{"type": "object", "properties": {}}}}\n'
    "</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)

# ---- 渲染 ----


def test_system_message_and_tools_share_the_system_block():
    prompt = chatml.render([Message("system", "你是助手"), Message("user", "谁最高？")], [BOARD, POLICIES])
    assert prompt == (
        f"<|im_start|>system\n你是助手\n\n{TOOL_BLOCK}<|im_end|>\n"
        "<|im_start|>user\n谁最高？<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_without_tools_there_is_no_tool_block():
    assert chatml.render([Message("system", "你是助手"), Message("user", "在吗")]) == (
        "<|im_start|>system\n你是助手<|im_end|>\n<|im_start|>user\n在吗<|im_end|>\n<|im_start|>assistant\n"
    )
    assert chatml.render([Message("user", "在吗")]) == "<|im_start|>user\n在吗<|im_end|>\n<|im_start|>assistant\n"


def test_tools_without_a_system_message_get_their_own_block():
    prompt = chatml.render([Message("user", "谁最高？")], [BOARD, POLICIES])
    assert prompt == (
        f"<|im_start|>system\n{TOOL_BLOCK}<|im_end|>\n<|im_start|>user\n谁最高？<|im_end|>\n<|im_start|>assistant\n"
    )


def test_prefix_is_written_after_the_assistant_header():
    assert chatml.render([Message("user", "在吗")], prefix="<think>\n").endswith("<|im_start|>assistant\n<think>\n")


HISTORY = [
    Message("system", "你是助手"),
    Message("user", "谁最高？"),
    Message(
        "assistant",
        "",
        tool_calls=(ToolCall("c1", "leaderboard", {"suite": "main"}), ToolCall("c2", "list_policies", {})),
    ),
    Message("tool", '{"rows": []}', tool_call_id="c1"),
    Message("tool", '{"policies": ["甲"]}', tool_call_id="c2"),
    Message("assistant", "还没有结果。"),
    Message("user", "那再看一次"),
    Message("assistant", "好，我查一下", tool_calls=(ToolCall("c3", "leaderboard", {}),)),
    Message("tool", "{}", tool_call_id="c3"),
]


def test_tool_calls_and_results_are_rendered_into_the_history():
    prompt = chatml.render(HISTORY, [BOARD, POLICIES])
    head, tail = prompt.split("<|im_start|>user\n谁最高？<|im_end|>\n", 1)
    assert head == f"<|im_start|>system\n你是助手\n\n{TOOL_BLOCK}<|im_end|>\n"
    assert tail == (
        # 一条助手消息里的几次调用各占一个 <tool_call> 块
        '<|im_start|>assistant\n<tool_call>\n{"name": "leaderboard", "arguments": {"suite": "main"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "list_policies", "arguments": {}}\n</tool_call><|im_end|>\n'
        # 连续的工具结果合在一条 user 消息里
        '<|im_start|>user\n<tool_response>\n{"rows": []}\n</tool_response>\n'
        '<tool_response>\n{"policies": ["甲"]}\n</tool_response><|im_end|>\n'
        "<|im_start|>assistant\n还没有结果。<|im_end|>\n"
        "<|im_start|>user\n那再看一次<|im_end|>\n"
        '<|im_start|>assistant\n好，我查一下\n<tool_call>\n{"name": "leaderboard", "arguments": {}}\n</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_a_malformed_call_is_rendered_the_way_the_model_wrote_it():
    """模型上一轮写坏了的调用要原样留在历史里，它才看得懂随后的报错。"""
    broken = (
        ToolCall("c1", "leaderboard", {}, raw_arguments='["main"]'),  # 有工具名，参数不是对象
        ToolCall("c2", "", {}, raw_arguments='{"name": "leaderboard", "arguments": {'),  # 整块解析不了
    )
    prompt = chatml.render([Message("user", "?"), Message("assistant", "", tool_calls=broken)])
    assert '<tool_call>\n{"name": "leaderboard", "arguments": ["main"]}\n</tool_call>' in prompt
    assert '<tool_call>\n{"name": "leaderboard", "arguments": {\n</tool_call><|im_end|>' in prompt


def test_a_later_system_message_is_an_ordinary_turn():
    prompt = chatml.render([Message("system", "甲"), Message("user", "一"), Message("system", "乙")])
    assert prompt == (
        "<|im_start|>system\n甲<|im_end|>\n<|im_start|>user\n一<|im_end|>\n"
        "<|im_start|>system\n乙<|im_end|>\n<|im_start|>assistant\n"
    )


def test_an_unknown_role_is_an_error():
    with pytest.raises(ValueError, match="不认识的消息角色：narrator"):
        chatml.render([Message("narrator", "很久以前")])


# ---- 解析 ----

CALL = '<tool_call>\n{"name": "leaderboard", "arguments": {"suite": "main"}}\n</tool_call>'
PARSED = ToolCall("call_0", "leaderboard", {"suite": "main"})


def test_a_single_call_is_parsed():
    assert chatml.parse(CALL) == ("", (PARSED,))


def test_plain_text_has_no_calls():
    assert chatml.parse("  最高的是 a。\n") == ("最高的是 a。", ())
    assert chatml.parse("") == ("", ())


def test_text_around_calls_is_kept_and_calls_are_numbered():
    second = '<tool_call>{"name": "list_policies", "arguments": {}}</tool_call>'
    text, calls = chatml.parse(f"我先查一下。\n{CALL}\n{second}\n稍等")
    assert text == "我先查一下。\n稍等"
    assert calls == (PARSED, ToolCall("call_1", "list_policies", {}))


def test_missing_or_null_arguments_mean_no_arguments():
    for body in ('{"name": "list_policies"}', '{"name": "list_policies", "arguments": null}'):
        assert chatml.parse(f"<tool_call>\n{body}\n</tool_call>") == ("", (ToolCall("call_0", "list_policies", {}),))


def test_reasoning_is_dropped():
    assert chatml.parse(f"<think>\n用户要看排行榜。\n</think>\n\n{CALL}") == ("", (PARSED,))
    assert chatml.parse("<think>\n想\n</think>\n\n最高的是 a") == ("最高的是 a", ())


def test_a_call_mentioned_while_reasoning_is_not_a_call():
    assert chatml.parse(f"<think>\n我该输出 {CALL} 吗？再想想。\n</think>\n\n不需要工具。") == ("不需要工具。", ())


def test_unfinished_reasoning_yields_nothing():
    """思考还没结束，输出就到了上限或被中止：后面的全是思考，里面提到的调用不算数。"""
    assert chatml.parse(f"<think>\n用户要看排行榜，我应该调用 {CALL}") == ("", ())
    assert chatml.parse("先说一句<think>\n然后一直想") == ("先说一句", ())


def test_a_closing_tag_alone_still_ends_the_reasoning():
    """开头的 <think> 写在提示里时，调用方应把它拼回来；没拼也以最后一个 </think> 为界。"""
    assert chatml.parse(f"想一想\n</think>\n\n{CALL}") == ("", (PARSED,))


@pytest.mark.parametrize(
    "body",
    [
        '{"name": "leaderboard", "arguments": {"suite": ',  # JSON 不完整
        "leaderboard(suite=main)",  # 不是 JSON
        '["leaderboard"]',  # 不是对象
        '{"arguments": {}}',  # 没有工具名
        '{"name": 3, "arguments": {}}',  # 工具名不是字符串
    ],
)
def test_an_unparseable_block_is_kept_verbatim_with_no_name(body):
    assert chatml.parse(f"<tool_call>\n{body}\n</tool_call>") == ("", (ToolCall("call_0", "", {}, raw_arguments=body),))


@pytest.mark.parametrize(
    "arguments",
    ['["main"]', '"{\\"suite\\": \\"main\\"}"', "3", "true"],
)
def test_arguments_that_are_not_an_object_are_kept_as_text(arguments):
    """参数必须是 JSON 对象。数组、字符串（哪怕里面是一段 JSON）、数值都不算，原文交给上层。"""
    (call,) = chatml.parse(f'<tool_call>\n{{"name": "leaderboard", "arguments": {arguments}}}\n</tool_call>')[1]
    assert (call.id, call.name, call.arguments, call.raw_arguments) == ("call_0", "leaderboard", {}, arguments)


def test_the_last_block_may_lack_its_closing_tag():
    assert chatml.parse('<tool_call>\n{"name": "list_policies", "arguments": {}}') == (
        "",
        (ToolCall("call_0", "list_policies", {}),),
    )
    # 输出被上限截断：JSON 不完整，原文留给上层
    cut = '{"name": "leaderboard", "argu'
    assert chatml.parse(f"<tool_call>\n{cut}") == ("", (ToolCall("call_0", "", {}, raw_arguments=cut),))


def test_bare_json_without_tags_is_not_a_call():
    text = '{"name": "leaderboard", "arguments": {"suite": "main"}}'
    assert chatml.parse(text) == (text, ())
