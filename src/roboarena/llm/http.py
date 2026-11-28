"""两个 HTTP 适配器共用的部分：建客户端、发请求、把失败统一成 ModelError。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import httpx

from roboarena.core.errors import ModelError

_BODY_CHARS = 200  # 报错里最多带多少个字符的服务端回应


def is_loopback(base_url: str) -> bool:
    return httpx.URL(base_url).host in ("127.0.0.1", "localhost", "::1")


def make_client(base_url: str, timeout_s: float, transport: httpx.BaseTransport | None) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
        transport=transport,
        # 环境里设了 http_proxy 时，发往本机服务的请求不能交给代理
        trust_env=not is_loopback(base_url),
    )


def post_json(
    client: httpx.Client,
    url: str,
    payload: Mapping[str, Any],
    *,
    name: str,
    base_url: str,
    timeout_s: float,
    headers: Mapping[str, str] | None = None,
) -> tuple[Any, float]:
    """发一个 JSON 请求，返回（回应的 JSON，耗时毫秒）。超时、连不上、5xx、429 标为可重试。"""
    start = time.perf_counter()
    try:
        response = client.post(url, json=dict(payload), headers=dict(headers or {}))
    except httpx.TimeoutException as exc:
        raise ModelError(f"模型 {name} 超时（{timeout_s:g} 秒内没有回应）", retryable=True) from exc
    except httpx.TransportError as exc:
        raise ModelError(
            f"连不上模型 {name} 的服务 {base_url}（{type(exc).__name__}）；本机服务请先启动", retryable=True
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000.0
    if response.status_code != 200:
        body = " ".join(response.text.split())[:_BODY_CHARS]
        raise ModelError(
            f"模型 {name} 的服务返回 {response.status_code}：{body}",
            retryable=response.status_code >= 500 or response.status_code == 429,
        )
    try:
        return response.json(), latency_ms
    except ValueError as exc:
        raise ModelError(f"模型 {name} 的服务返回的内容不是预期的格式：不是 JSON") from exc
