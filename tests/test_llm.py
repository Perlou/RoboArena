"""模型网关：接口的数据类型、脚本化假模型。不联网，不调用真实模型。"""

import pytest

from roboarena.core.errors import ModelError, RoboArenaError
from roboarena.llm.api import ChatModel, Completion, Message, ToolCall, ToolSpec
from roboarena.llm.scripted import ScriptedModel

LEADERBOARD = ToolSpec("leaderboard", "查看排行榜", {"type": "object", "properties": {}})


def test_scripted_model_replies_in_order():
    call = ToolCall("c1", "leaderboard", {"suite": "main"})
    model = ScriptedModel("fake", [call, "最高的是 a"])
    asked = [Message("user", "谁最高？")]

    first = model.complete(asked, [LEADERBOARD])
    second = model.complete([*asked, Message("tool", "{}", tool_call_id="c1")], [LEADERBOARD])

    assert (first.text, first.tool_calls) == ("", (call,))
    assert (second.text, second.tool_calls) == ("最高的是 a", ())
    assert not first.cached and first.latency_ms == 0.0


def test_scripted_model_counts_tokens_from_text_length():
    done = ScriptedModel("fake", ["好的"]).complete([Message("system", "你是助手"), Message("user", "在吗")])
    assert (done.tokens_in, done.tokens_out) == (6, 2)


def test_scripted_model_records_what_it_was_asked():
    model = ScriptedModel("fake", ["甲", "乙"])
    model.complete([Message("user", "一")], [LEADERBOARD], seed=3)
    model.complete([Message("user", "二")])
    assert [r.messages[-1].content for r in model.requests] == ["一", "二"]
    assert model.requests[0].tools == (LEADERBOARD,) and model.requests[0].seed == 3


def test_scripted_model_can_answer_with_a_function():
    model = ScriptedModel("echo", lambda messages, tools: "收到：" + messages[-1].content)
    assert model.complete([Message("user", "你好")]).text == "收到：你好"
    assert model.complete([Message("user", "再见")]).text == "收到：再见"


def test_scripted_model_passes_a_full_completion_through():
    ready = Completion("嗯", (), tokens_in=7, tokens_out=1, latency_ms=12.5, cached=False)
    assert ScriptedModel("fake", [ready]).complete([Message("user", "?")]) is ready


def test_scripted_model_fails_clearly_when_the_script_runs_out():
    model = ScriptedModel("fake", ["只有一句"])
    model.complete([Message("user", "一")])
    with pytest.raises(ModelError, match="fake 的脚本只有 1 条回答"):
        model.complete([Message("user", "二")])
    assert issubclass(ModelError, RoboArenaError)


def test_scripted_model_is_a_chat_model():
    model: ChatModel = ScriptedModel("fake", [])
    assert model.name == "fake"


# ---- OpenAI 兼容适配器：用 httpx 的 MockTransport，不联网 ----

import json  # noqa: E402

import httpx  # noqa: E402

from roboarena.config import LLMConfig, LLMModelConfig  # noqa: E402
from roboarena.core.errors import NotFoundError  # noqa: E402
from roboarena.llm.gateway import Gateway, read_calls, request_key  # noqa: E402
from roboarena.llm.openai_compat import OpenAIChatModel  # noqa: E402

REMOTE = LLMModelConfig(base_url="http://model.test/v1/", model="m-1", api_key_env="ROBOARENA_TEST_KEY")


def _reply(message: dict, usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3} if usage is None else usage,
    }


def _adapter(handler, cfg: LLMModelConfig = REMOTE, **kwargs) -> OpenAIChatModel:
    return OpenAIChatModel("remote", cfg, transport=httpx.MockTransport(handler), **kwargs)


def test_adapter_sends_an_openai_request_and_parses_the_answer(monkeypatch):
    monkeypatch.setenv("ROBOARENA_TEST_KEY", "sk-secret")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        call = {
            "id": "c9",
            "type": "function",
            "function": {"name": "leaderboard", "arguments": '{"suite": "main"}'},
        }
        return httpx.Response(200, json=_reply({"role": "assistant", "content": None, "tool_calls": [call]}))

    history = [
        Message("system", "你是助手"),
        Message("user", "谁最高？"),
        Message("assistant", "", tool_calls=(ToolCall("c1", "list_policies", {"task": "toy"}),)),
        Message("tool", '{"policies": []}', tool_call_id="c1"),
    ]
    done = _adapter(handler).complete(history, [LEADERBOARD], temperature=0.0, max_tokens=64, seed=7)

    assert done.tool_calls == (ToolCall("c9", "leaderboard", {"suite": "main"}),)
    assert (done.text, done.tokens_in, done.tokens_out, done.cached) == ("", 11, 3, False)
    assert done.latency_ms >= 0.0
    (request,) = seen
    assert str(request.url) == "http://model.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-secret"
    body = json.loads(request.content)
    assert (body["model"], body["seed"], body["max_tokens"], body["stream"]) == ("m-1", 7, 64, False)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "leaderboard",
                "description": "查看排行榜",
                "parameters": LEADERBOARD.parameters,
            },
        }
    ]
    assert body["messages"][2]["tool_calls"][0]["function"] == {
        "name": "list_policies",
        "arguments": '{"task": "toy"}',
    }
    assert body["messages"][3] == {"role": "tool", "content": '{"policies": []}', "tool_call_id": "c1"}


def test_adapter_omits_tools_and_auth_when_there_are_none():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_reply({"role": "assistant", "content": "好"}, usage={}))

    done = _adapter(handler, LLMModelConfig(model="m")).complete([Message("user", "在吗")])
    assert (done.text, done.tool_calls, done.tokens_in) == ("好", (), 0)
    assert "authorization" not in seen[0].headers and "tools" not in json.loads(seen[0].content)


@pytest.mark.parametrize("raw", ['{"suite": ', '["main"]', "main"])
def test_adapter_keeps_unparseable_arguments_for_the_caller(raw):
    def handler(request: httpx.Request) -> httpx.Response:
        call = {"function": {"name": "leaderboard", "arguments": raw}}
        return httpx.Response(200, json=_reply({"content": "", "tool_calls": [call]}))

    (call,) = _adapter(handler, LLMModelConfig(model="m")).complete([Message("user", "?")]).tool_calls
    assert (call.id, call.name, call.arguments, call.raw_arguments) == ("call_0", "leaderboard", {}, raw)


def test_adapter_accepts_arguments_given_as_an_object_or_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        calls = [
            {"id": "a", "function": {"name": "x", "arguments": {"n": 1}}},
            {"id": "b", "function": {"name": "y", "arguments": ""}},
        ]
        return httpx.Response(200, json=_reply({"content": "", "tool_calls": calls}))

    done = _adapter(handler, LLMModelConfig(model="m")).complete([Message("user", "?")])
    assert done.tool_calls == (ToolCall("a", "x", {"n": 1}), ToolCall("b", "y", {}))


def _raises_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("慢", request=request)


def _raises_refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("拒绝", request=request)


@pytest.mark.parametrize(
    "handler, retryable, message",
    [
        (_raises_timeout, True, "模型 remote 超时（5 秒内没有回应）"),
        (_raises_refused, True, "连不上模型 remote 的服务 http://127.0.0.1:11434/v1"),
        (lambda r: httpx.Response(503, text="忙"), True, "返回 503：忙"),
        (lambda r: httpx.Response(429, text="慢点"), True, "返回 429"),
        (
            lambda r: httpx.Response(404, json={"error": "model 'm' not found"}),
            False,
            "返回 404：.*not found",
        ),
        (lambda r: httpx.Response(200, text="<html>"), False, "不是预期的格式"),
        (lambda r: httpx.Response(200, json={"choices": []}), False, "不是预期的格式"),
    ],
)
def test_adapter_turns_failures_into_model_errors(handler, retryable, message):
    model = _adapter(handler, LLMModelConfig(model="m"), timeout_s=5)
    with pytest.raises(ModelError, match=message) as info:
        model.complete([Message("user", "?")])
    assert info.value.retryable is retryable
    assert "\n" not in str(info.value)


def test_adapter_reports_a_missing_key_without_calling_the_server(monkeypatch):
    monkeypatch.delenv("ROBOARENA_TEST_KEY", raising=False)
    model = _adapter(lambda r: pytest.fail("不该发出请求"))
    with pytest.raises(ModelError, match="环境变量 ROBOARENA_TEST_KEY 里，但它没有设置") as info:
        model.complete([Message("user", "?")])
    assert not info.value.retryable


# ---- 网关 ----


class Flaky:
    """前 n 次调用失败，之后交给里面的假模型。"""

    def __init__(self, name: str, failures: int, inner: ScriptedModel, *, retryable: bool = True) -> None:
        self.name, self.failures, self.inner, self.retryable = name, failures, inner, retryable
        self.calls = 0

    def complete(self, messages, tools=(), **params) -> Completion:
        self.calls += 1
        if self.calls <= self.failures:
            raise ModelError(f"模型 {self.name} 超时", retryable=self.retryable)
        return self.inner.complete(messages, tools, **params)


def _llm(**kwargs) -> LLMConfig:
    models = {
        "big": LLMModelConfig(model="m-big", price_in=2.0, price_out=6.0),
        "small": LLMModelConfig(model="m-small"),
    }
    return LLMConfig(**{"default": "big", "models": models, **kwargs})


ASK = [Message("user", "谁最高？")]


def test_gateway_routes_by_name_and_defaults_to_the_configured_model(tmp_path):
    big, small = ScriptedModel("big", ["大"]), ScriptedModel("small", ["小"])
    gateway = Gateway(_llm(), models={"big": big, "small": small})
    assert gateway.complete(ASK).text == "大"
    assert gateway.complete(ASK, model="small", seed=4).text == "小"
    assert small.requests[0].seed == 4
    with pytest.raises(NotFoundError, match="没有名为 nope 的模型；配置里定义的有：big、small"):
        gateway.complete(ASK, model="nope")


def test_gateway_retries_then_succeeds(tmp_path):
    flaky = Flaky("big", 2, ScriptedModel("big", ["好了"]))
    waits: list[float] = []
    calls = tmp_path / "llm_calls.jsonl"
    gateway = Gateway(_llm(retries=2), models={"big": flaky}, calls_path=calls, sleep=waits.append)

    assert gateway.complete(ASK).text == "好了"
    assert flaky.calls == 3 and waits == [0.5, 1.0]
    lines = read_calls(calls)
    assert [(c["attempt"], c["ok"]) for c in lines] == [(1, False), (2, False), (3, True)]
    assert lines[0]["error"] == "模型 big 超时" and "error" not in lines[2]


def test_gateway_gives_up_with_one_line_when_there_is_no_fallback():
    flaky = Flaky("big", 99, ScriptedModel("big", []))
    gateway = Gateway(_llm(retries=1), models={"big": flaky}, sleep=lambda s: None)
    with pytest.raises(ModelError, match=r"^模型 big 超时（共试 2 次）$"):
        gateway.complete(ASK)
    assert flaky.calls == 2


def test_gateway_does_not_retry_errors_that_would_repeat():
    flaky = Flaky("big", 99, ScriptedModel("big", []), retryable=False)
    gateway = Gateway(_llm(retries=3), models={"big": flaky}, sleep=lambda s: pytest.fail("不该等待"))
    with pytest.raises(ModelError, match="共试 1 次"):
        gateway.complete(ASK)
    assert flaky.calls == 1


def test_gateway_falls_back_and_says_so_in_the_record(tmp_path):
    flaky = Flaky("big", 99, ScriptedModel("big", []))
    small = ScriptedModel("small", ["小的来答"])
    calls = tmp_path / "llm_calls.jsonl"
    gateway = Gateway(
        _llm(retries=1, fallback="small"),
        models={"big": flaky, "small": small},
        calls_path=calls,
        sleep=lambda s: None,
    )
    assert gateway.complete(ASK).text == "小的来答"
    last = read_calls(calls)[-1]
    assert (last["model"], last["fallback_from"], last["ok"]) == ("small", "big", True)


def test_gateway_reports_both_failures_when_the_fallback_fails_too():
    models = {
        "big": Flaky("big", 99, ScriptedModel("big", [])),
        "small": Flaky("small", 99, ScriptedModel("small", [])),
    }
    gateway = Gateway(_llm(retries=0, fallback="small"), models=models, sleep=lambda s: None)
    with pytest.raises(
        ModelError, match="模型 big 超时（共试 1 次）；备用模型也失败：模型 small 超时"
    ) as info:
        gateway.complete(ASK)
    assert "\n" not in str(info.value)


def test_gateway_caches_identical_requests(tmp_path):
    call = ToolCall("c1", "leaderboard", {"suite": "main"})
    big = ScriptedModel("big", [call, "第二个问题的回答", "换了种子"])
    calls = tmp_path / "llm_calls.jsonl"
    gateway = Gateway(_llm(), cache_dir=tmp_path / "cache", models={"big": big}, calls_path=calls)

    first = gateway.complete(ASK, [LEADERBOARD])
    again = gateway.complete(ASK, [LEADERBOARD])
    assert again.tool_calls == first.tool_calls == (call,)
    assert (first.cached, again.cached, again.latency_ms) == (False, True, 0.0)
    assert (again.tokens_in, again.tokens_out) == (first.tokens_in, first.tokens_out)
    assert len(big.requests) == 1

    gateway.complete([Message("user", "另一个问题")], [LEADERBOARD])
    gateway.complete(ASK, [LEADERBOARD], seed=1)
    assert len(big.requests) == 3  # 消息或种子不同就不是同一个请求
    assert [c["cached"] for c in read_calls(calls)] == [False, True, False, False]

    # 另一个进程（新的网关对象）也能用上
    fresh = Gateway(_llm(), cache_dir=tmp_path / "cache", models={"big": ScriptedModel("big", [])})
    assert fresh.complete(ASK, [LEADERBOARD]).cached


def test_gateway_cache_can_be_turned_off_and_survives_a_broken_file(tmp_path):
    big = ScriptedModel("big", ["一", "二", "三", "四"])
    off = Gateway(_llm(cache=False), cache_dir=tmp_path / "cache", models={"big": big})
    assert [off.complete(ASK).text for _ in range(2)] == ["一", "二"]
    assert not (tmp_path / "cache").exists()

    on = Gateway(_llm(), cache_dir=tmp_path / "cache", models={"big": big})
    assert on.complete(ASK).text == "三"
    (path,) = (tmp_path / "cache").rglob("*.json")
    path.write_text("{坏了", encoding="utf-8")
    assert on.complete(ASK).text == "四"


def test_cache_key_depends_on_every_part_of_the_request():
    base = {"temperature": 0.0, "max_tokens": 512, "seed": 0}
    key = request_key("m", ASK, [LEADERBOARD], **base)
    assert key == request_key("m", list(ASK), (LEADERBOARD,), **base)
    others = [
        request_key("m2", ASK, [LEADERBOARD], **base),
        request_key("m", [Message("user", "谁最低？")], [LEADERBOARD], **base),
        request_key("m", ASK, [], **base),
        request_key("m", ASK, [LEADERBOARD], **{**base, "temperature": 0.7}),
        request_key("m", ASK, [LEADERBOARD], **{**base, "max_tokens": 64}),
        request_key("m", ASK, [LEADERBOARD], **{**base, "seed": 1}),
    ]
    assert len({key, *others}) == 7


def test_call_records_carry_usage_and_cost(tmp_path):
    ready = Completion("嗯", (), tokens_in=1_000_000, tokens_out=500_000, latency_ms=12.5, cached=False)
    calls = tmp_path / "llm_calls.jsonl"
    gateway = Gateway(
        _llm(), cache_dir=tmp_path / "cache", models={"big": ScriptedModel("big", [ready])}, calls_path=calls
    )
    gateway.complete(ASK)
    gateway.complete(ASK)
    first, second = read_calls(calls)
    assert (first["model"], first["model_id"], first["latency_ms"]) == ("big", "m-big", 12.5)
    assert (first["tokens_in"], first["tokens_out"], first["cost"]) == (
        1_000_000,
        500_000,
        5.0,
    )  # 2 + 0.5 × 6
    assert (second["cached"], second["cost"], second["latency_ms"]) == (True, 0.0, 0.0)
    assert first["request"] == second["request"] and len(first["request"]) == 64


def test_the_key_never_reaches_records_cache_or_errors(tmp_path, monkeypatch):
    """FR-L6：密钥只在请求头里出现。"""
    monkeypatch.setenv("ROBOARENA_TEST_KEY", "sk-very-secret")
    answers = iter([httpx.Response(500, text="坏了"), httpx.Response(200, json=_reply({"content": "好"}))])
    adapter = _adapter(lambda r: next(answers))
    cfg = LLMConfig(default="remote", retries=1, models={"remote": REMOTE})
    calls = tmp_path / "llm_calls.jsonl"
    gateway = Gateway(
        cfg, cache_dir=tmp_path / "cache", models={"remote": adapter}, calls_path=calls, sleep=lambda s: None
    )

    assert gateway.complete(ASK).text == "好"
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert "llm_calls" in str(calls) and "坏了" in written
    assert "sk-very-secret" not in written

    from roboarena.config import Config

    snapshot = Config(llm=cfg).to_toml()
    assert "ROBOARENA_TEST_KEY" in snapshot and "sk-very-secret" not in snapshot


@pytest.mark.parametrize(
    "base_url, uses_proxy",
    [
        ("http://127.0.0.1:11434/v1", False),
        ("http://localhost:8000/v1", False),
        ("http://[::1]:8000/v1", False),
        ("https://api.example.com/v1", True),
    ],
)
def test_local_servers_are_not_reached_through_the_proxy(monkeypatch, base_url, uses_proxy):
    """环境里设了代理时，发往本机服务的请求不能交给代理（否则连不上）。"""
    from roboarena.llm.http import is_loopback

    monkeypatch.setenv("http_proxy", "http://proxy.test:7890")
    monkeypatch.setenv("all_proxy", "http://proxy.test:7890")
    assert is_loopback(base_url) is not uses_proxy
    model = OpenAIChatModel("m", LLMModelConfig(base_url=base_url, model="m"))
    assert model._client.trust_env is uses_proxy


# ---- 原样补全的适配器（Ollama 的 /api/generate，raw）：提示由项目自己渲染，输出由项目自己解析 ----

from roboarena.llm import chatml  # noqa: E402
from roboarena.llm.ollama_raw import OllamaRawModel  # noqa: E402

LOCAL = LLMModelConfig(backend="ollama", base_url="http://127.0.0.1:11434/", model="m-raw")


def _raw(handler, cfg: LLMModelConfig = LOCAL, **kwargs) -> OllamaRawModel:
    return OllamaRawModel("local", cfg, transport=httpx.MockTransport(handler), **kwargs)


def _generated(response: str, **fields) -> dict:
    return {
        "response": response,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 40,
        "eval_count": 9,
        **fields,
    }


def test_raw_adapter_sends_its_own_prompt_and_parses_the_output():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        text = '<tool_call>\n{"name": "leaderboard", "arguments": {"suite": "main"}}\n</tool_call>'
        return httpx.Response(200, json=_generated(text))

    asked = [Message("system", "你是助手"), Message("user", "谁最高？")]
    done = _raw(handler).complete(asked, [LEADERBOARD], temperature=0.0, max_tokens=64, seed=7)

    assert done.tool_calls == (ToolCall("call_0", "leaderboard", {"suite": "main"}),)
    assert (done.text, done.tokens_in, done.tokens_out, done.cached, done.finish_reason) == (
        "",
        40,
        9,
        False,
        "stop",
    )
    (request,) = seen
    assert str(request.url) == "http://127.0.0.1:11434/api/generate"
    body = json.loads(request.content)
    assert body == {
        "model": "m-raw",
        "prompt": chatml.render(asked, [LEADERBOARD]),
        "raw": True,  # 不让服务端套它自己的模板
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64, "seed": 7},
    }


def test_raw_adapter_writes_the_prefix_and_reads_the_output_as_its_continuation():
    """带思考的模型：助手开头的 <think> 写在提示里，解析时要拼回来，思考的内容才会被丢掉。"""
    thinking = LLMModelConfig(backend="ollama", model="m-think", assistant_prefix="<think>\n")
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["prompt"])
        return httpx.Response(
            200, json=_generated("要不要调用 <tool_call>…</tool_call>？不用。\n</think>\n\n最高的是 a")
        )

    done = _raw(handler, thinking).complete([Message("user", "谁最高？")])
    assert prompts[0].endswith("<|im_start|>assistant\n<think>\n")
    assert (done.text, done.tool_calls) == ("最高的是 a", ())


@pytest.mark.parametrize(
    "reply, finish, tokens",
    [
        (_generated("好"), "stop", (40, 9)),
        (_generated("好", done_reason="length"), "length", (40, 9)),
        # 服务端中止（如同一个 token 重复太多次）：done 为 false，没有结束原因，也没有 token 数
        ({"response": "NNNNNN", "done": False}, "aborted", (0, 0)),
        ({"response": "好", "done": True, "done_reason": "unload"}, "unload", (0, 0)),
    ],
)
def test_raw_adapter_reports_why_generation_ended(reply, finish, tokens):
    done = _raw(lambda r: httpx.Response(200, json=reply)).complete([Message("user", "?")])
    assert (done.finish_reason, (done.tokens_in, done.tokens_out)) == (finish, tokens)


@pytest.mark.parametrize(
    "handler, retryable, message",
    [
        (_raises_timeout, True, "模型 local 超时（5 秒内没有回应）"),
        (_raises_refused, True, "连不上模型 local 的服务 http://127.0.0.1:11434/"),
        (lambda r: httpx.Response(500, text="坏了"), True, "返回 500：坏了"),
        (
            lambda r: httpx.Response(404, json={"error": "model 'm-raw' not found"}),
            False,
            "返回 404：.*not found",
        ),
        (lambda r: httpx.Response(200, text="<html>"), False, "不是预期的格式"),
        (lambda r: httpx.Response(200, json={"done": True}), False, "不是预期的格式：缺少字段 'response'"),
    ],
)
def test_raw_adapter_turns_failures_into_model_errors(handler, retryable, message):
    with pytest.raises(ModelError, match=message) as info:
        _raw(handler, timeout_s=5).complete([Message("user", "?")])
    assert info.value.retryable is retryable and "\n" not in str(info.value)


def test_raw_adapter_does_not_use_the_proxy_for_a_local_server(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.test:7890")
    assert OllamaRawModel("local", LOCAL)._client.trust_env is False


# ---- 结束原因 ----


def test_openai_adapter_reports_why_generation_ended():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "半句"}, "finish_reason": "length"}]}
        )

    assert (
        _adapter(handler, LLMModelConfig(model="m")).complete([Message("user", "?")]).finish_reason
        == "length"
    )


def test_finish_reason_defaults_to_stop_and_survives_the_cache(tmp_path):
    assert ScriptedModel("fake", ["好"]).complete([Message("user", "?")]).finish_reason == "stop"
    cut = Completion(
        "半", (), tokens_in=3, tokens_out=512, latency_ms=1.0, cached=False, finish_reason="length"
    )
    gateway = Gateway(_llm(), cache_dir=tmp_path / "cache", models={"big": ScriptedModel("big", [cut])})
    assert [gateway.complete(ASK).finish_reason for _ in range(2)] == ["length", "length"]


# ---- 网关按后端建适配器；模型的身份 ----


def test_gateway_builds_the_adapter_named_by_the_backend():
    cfg = LLMConfig(default="a", models={"a": LLMModelConfig(backend="openai", model="m"), "b": LOCAL})
    gateway = Gateway(cfg)
    assert type(gateway._model("a")) is OpenAIChatModel and type(gateway._model("b")) is OllamaRawModel


def test_model_identity_covers_everything_that_changes_the_answer():
    base = LLMModelConfig(backend="ollama", base_url="http://127.0.0.1:11434", model="m")
    variants = {
        "base": base,
        "same": LLMModelConfig(backend="ollama", base_url="http://127.0.0.1:11434", model="m", price_in=9.0),
        "backend": LLMModelConfig(backend="openai", base_url="http://127.0.0.1:11434", model="m"),
        "url": LLMModelConfig(backend="ollama", base_url="http://127.0.0.1:11435", model="m"),
        "model": LLMModelConfig(backend="ollama", base_url="http://127.0.0.1:11434", model="m2"),
        "prefix": LLMModelConfig(
            backend="ollama", base_url="http://127.0.0.1:11434", model="m", assistant_prefix="<think>\n"
        ),
    }
    gateway = Gateway(LLMConfig(default="base", models=variants))
    ids = {name: gateway.model_id(name) for name in variants}
    assert ids["base"] == ids["same"]  # 价格不改变回答
    assert len({ids[n] for n in ("base", "backend", "url", "model", "prefix")}) == 5


# ---- 上下文长度：显式写进请求，不依赖服务的默认值 ----


def test_raw_adapter_sends_the_context_length_when_configured():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_generated("好"))

    sized = LLMModelConfig(backend="ollama", model="m", num_ctx=8192)
    _raw(handler, sized).complete([Message("user", "?")], max_tokens=64, seed=1)
    _raw(handler, LOCAL).complete([Message("user", "?")], max_tokens=64, seed=1)
    assert bodies[0]["options"] == {"temperature": 0.0, "num_predict": 64, "seed": 1, "num_ctx": 8192}
    assert "num_ctx" not in bodies[1]["options"]  # 0 表示用服务的默认值


def test_context_length_is_part_of_the_model_identity():
    """上下文不够时服务端会悄悄截掉提示的开头，回答就变了，所以它算进模型的身份。"""
    small = LLMModelConfig(backend="ollama", model="m")
    large = LLMModelConfig(backend="ollama", model="m", num_ctx=8192)
    gateway = Gateway(LLMConfig(default="small", models={"small": small, "large": large}))
    assert gateway.model_id("small") != gateway.model_id("large")
    assert gateway.model_id("large").endswith("|ctx=8192")
