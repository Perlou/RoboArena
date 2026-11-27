"""OpenAI 兼容的 HTTP 接口（/chat/completions）。Ollama、vLLM、SGLang、llama.cpp 的服务都提供这个接口。"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from roboarena.config import LLMModelConfig
from roboarena.core.errors import ModelError
from roboarena.llm.api import Completion, Message, ToolCall, ToolSpec

_BODY_CHARS = 200  # 报错里最多带多少个字符的服务端回应


def is_loopback(base_url: str) -> bool:
    return httpx.URL(base_url).host in ("127.0.0.1", "localhost", "::1")


def _message_payload(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    return out


def build_payload(
    model: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [_message_payload(m) for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]
    return payload


def _parse_tool_call(index: int, raw: Any) -> ToolCall:
    function = raw.get("function") or {}
    name = function.get("name")
    if not isinstance(name, str):
        raise ValueError("工具调用缺少名称")
    call_id = raw.get("id") or f"call_{index}"
    arguments = function.get("arguments")
    if isinstance(arguments, dict):  # 有的服务直接给对象
        return ToolCall(call_id, name, arguments)
    text = "" if arguments is None else str(arguments)
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return ToolCall(call_id, name, {}, raw_arguments=text)
    return ToolCall(call_id, name, parsed)


def parse_response(data: Any, latency_ms: float) -> Completion:
    """把 /chat/completions 的回应变成 Completion。结构不对时抛 ValueError。"""
    try:
        message = data["choices"][0]["message"]
        calls = tuple(_parse_tool_call(i, raw) for i, raw in enumerate(message.get("tool_calls") or ()))
        usage = data.get("usage") or {}
        return Completion(
            text=message.get("content") or "",
            tool_calls=calls,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            cached=False,
        )
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ValueError(f"缺少字段 {exc}") from exc


class OpenAIChatModel:
    """一个模型对应一个实例。密钥在每次请求时从环境变量读，不保存在对象里，也不出现在报错里。"""

    def __init__(
        self,
        name: str,
        cfg: LLMModelConfig,
        *,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = name
        self._cfg = cfg
        self._timeout_s = timeout_s
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
            # 环境里设了 http_proxy 时，发往本机服务的请求不能交给代理
            trust_env=not is_loopback(cfg.base_url),
        )

    def _headers(self) -> dict[str, str]:
        env = self._cfg.api_key_env
        if not env:
            return {}
        key = os.environ.get(env, "")
        if not key:
            raise ModelError(f"模型 {self.name} 的密钥应放在环境变量 {env} 里，但它没有设置")
        return {"Authorization": f"Bearer {key}"}

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Completion:
        payload = build_payload(
            self._cfg.model, messages, tools, temperature=temperature, max_tokens=max_tokens, seed=seed
        )
        headers = self._headers()
        start = time.perf_counter()
        try:
            response = self._client.post(self._url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise ModelError(
                f"模型 {self.name} 超时（{self._timeout_s:g} 秒内没有回应）", retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise ModelError(
                f"连不上模型 {self.name} 的服务 {self._cfg.base_url}（{type(exc).__name__}）；本机服务请先启动",
                retryable=True,
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        if response.status_code != 200:
            body = " ".join(response.text.split())[:_BODY_CHARS]
            raise ModelError(
                f"模型 {self.name} 的服务返回 {response.status_code}：{body}",
                retryable=response.status_code >= 500 or response.status_code == 429,
            )
        try:
            return parse_response(response.json(), latency_ms)
        except ValueError as exc:  # 含 JSON 解析失败
            raise ModelError(f"模型 {self.name} 的服务返回的内容不是预期的格式：{exc}") from exc
