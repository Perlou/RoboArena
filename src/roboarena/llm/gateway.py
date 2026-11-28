"""模型网关：按名称取模型，重试与降级，缓存，调用记录（docs/ARCHITECTURE.md 16.2 节）。

Agent 与评测只和 Gateway 打交道。它不知道工作区的存在：缓存目录与调用记录的位置由调用方给。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboarena.config import LLMConfig
from roboarena.core.errors import ModelError, NotFoundError
from roboarena.core.hashing import canonical_json, sha256_text
from roboarena.core.log import get_logger
from roboarena.llm.api import ChatModel, Completion, Message, ToolCall, ToolSpec
from roboarena.llm.ollama_raw import OllamaRawModel
from roboarena.llm.openai_compat import OpenAIChatModel

log = get_logger(__name__)

BACKOFF_S = 0.5  # 第 k 次重试前等 BACKOFF_S * 2**(k-1) 秒


def request_key(
    model_id: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> str:
    """请求的摘要，也是缓存的键：模型标识、消息、工具、采样参数、种子的规范化 JSON 的 SHA-256。"""
    return sha256_text(
        canonical_json(
            {
                "model": model_id,
                "messages": [asdict(m) for m in messages],
                "tools": [asdict(t) for t in tools],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
    )


class Gateway:
    """models 用来注入现成的模型（测试里的假模型）；没注入的名称按配置建适配器。

    calls_path 是调用记录的位置（所属运行目录里的 llm_calls.jsonl），可以随时换；None 表示不记。
    cache_dir 为 None 或配置里 cache = false 时不用缓存。
    """

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        cache_dir: Path | None = None,
        calls_path: Path | None = None,
        models: Mapping[str, ChatModel] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.calls_path = calls_path
        self._cache_dir = cache_dir if cfg.cache else None
        self._models: dict[str, ChatModel] = dict(models or {})
        self._sleep = sleep

    def _model(self, name: str) -> ChatModel:
        if name not in self.cfg.models:
            known = "、".join(self.cfg.models) or "（空）"
            raise NotFoundError(f"没有名为 {name} 的模型；配置里定义的有：{known}")
        if name not in self._models:
            mc = self.cfg.models[name]
            adapter = OllamaRawModel if mc.backend == "ollama" else OpenAIChatModel
            self._models[name] = adapter(name, mc, timeout_s=self.cfg.timeout_s)
        return self._models[name]

    def model_id(self, name: str) -> str:
        """模型的身份：会改变回答的那几项配置。缓存的键与 AI 评测的配置标识都用它。

        同一个模型标识在不同的服务上可能是不同的权重，所以地址也算进去；后端决定提示怎么渲染。
        """
        mc = self.cfg.models[name]
        parts = [mc.backend, mc.base_url, mc.model]
        if mc.assistant_prefix:
            parts.append(mc.assistant_prefix)
        if mc.num_ctx:  # 上下文不够时服务端会截掉提示的开头，回答就变了
            parts.append(f"ctx={mc.num_ctx}")
        return "|".join(parts)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Completion:
        """model 省略时用配置里的默认模型。全部尝试都失败时抛 ModelError（一行中文）。"""
        name = model or self.cfg.default
        self._model(name)  # 名称不对立刻报错，不走重试
        params: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        try:
            return self._complete_with(name, messages, tools, params, fallback_from=None)
        except ModelError as primary:
            backup = self.cfg.fallback
            if not backup or backup == name:
                raise
            log.warning("改用备用模型", extra={"model": name, "fallback": backup})
            try:
                return self._complete_with(backup, messages, tools, params, fallback_from=name)
            except ModelError as secondary:
                raise ModelError(f"{primary}；备用模型也失败：{secondary}") from secondary

    def _complete_with(
        self,
        name: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        params: dict[str, Any],
        *,
        fallback_from: str | None,
    ) -> Completion:
        chat = self._model(name)
        key = request_key(self.model_id(name), messages, tools, **params)
        base = {"model": name, "request": key, "fallback_from": fallback_from}

        hit = self._cache_get(key)
        if hit is not None:
            self._record({**base, "attempt": 0, "ok": True, "cached": True}, hit, name)
            return hit

        attempts = self.cfg.retries + 1
        last: ModelError | None = None
        tried = 0
        for attempt in range(1, attempts + 1):
            tried = attempt
            start = time.perf_counter()
            try:
                done = chat.complete(messages, tools, **params)
            except ModelError as exc:
                last = exc
                elapsed = (time.perf_counter() - start) * 1000.0
                self._record(
                    {**base, "attempt": attempt, "ok": False, "cached": False, "error": str(exc)},
                    None,
                    name,
                    latency_ms=elapsed,
                )
                if not exc.retryable or attempt == attempts:
                    break
                log.warning(
                    "模型调用失败，稍后重试", extra={"model": name, "attempt": attempt, "error": str(exc)}
                )
                self._sleep(BACKOFF_S * 2 ** (attempt - 1))
                continue
            self._cache_put(key, done)
            self._record({**base, "attempt": attempt, "ok": True, "cached": False}, done, name)
            return done
        assert last is not None
        raise ModelError(f"{last}（共试 {tried} 次）", retryable=last.retryable) from last

    # ---- 缓存 ----

    def _cache_file(self, key: str) -> Path | None:
        return None if self._cache_dir is None else self._cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Completion | None:
        path = self._cache_file(key)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Completion(
                text=data["text"],
                tool_calls=tuple(ToolCall(**c) for c in data["tool_calls"]),
                tokens_in=data["tokens_in"],
                tokens_out=data["tokens_out"],
                latency_ms=0.0,  # 这次没有调用模型；原来的耗时在第一次调用的记录里
                cached=True,
                finish_reason=data.get("finish_reason", "stop"),  # 早先写下的缓存没有这一项
            )
        except (ValueError, KeyError, TypeError):
            log.warning("缓存文件读不出来，当作没有命中", extra={"request": key})
            return None

    def _cache_put(self, key: str, done: Completion) -> None:
        path = self._cache_file(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "text": done.text,
            "tool_calls": [asdict(c) for c in done.tool_calls],
            "tokens_in": done.tokens_in,
            "tokens_out": done.tokens_out,
            "finish_reason": done.finish_reason,
        }
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # 写到一半被打断也不会留下半个文件

    # ---- 调用记录 ----

    def _record(
        self, entry: dict[str, Any], done: Completion | None, name: str, *, latency_ms: float | None = None
    ) -> None:
        if self.calls_path is None:
            return
        mc = self.cfg.models[name]
        tokens_in = done.tokens_in if done else 0
        tokens_out = done.tokens_out if done else 0
        spent = done is not None and not done.cached
        line = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            **entry,
            "model_id": mc.model,
            "latency_ms": round(done.latency_ms if done else latency_ms or 0.0, 3),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "finish_reason": done.finish_reason if done else None,
            # 按配置的单价估算；命中缓存的这一次没有花钱，记 0
            "cost": (tokens_in * mc.price_in + tokens_out * mc.price_out) / 1e6 if spent else 0.0,
        }
        with open(self.calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_calls(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
