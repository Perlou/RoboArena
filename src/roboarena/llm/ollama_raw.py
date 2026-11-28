"""Ollama 的原样补全（/api/generate，raw）：提示由项目自己渲染，输出由项目自己解析（llm/chatml.py）。

为什么不走 OpenAI 兼容接口：那条路上提示的渲染与输出的解析都在服务端，各模型各一套。
实测（Ollama 0.17.5）它把一个模型的工具描述渲染成了不是 JSON 的东西，还会把参数形状不对的调用整个吞掉；
它的 /v1/completions 也会套对话模板，不是原样补全。详见 docs/PROGRESS.md 的 A1 明细。
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from roboarena.config import LLMModelConfig
from roboarena.core.errors import ModelError
from roboarena.llm import chatml
from roboarena.llm.api import Completion, Message, ToolSpec
from roboarena.llm.http import make_client, post_json


class OllamaRawModel:
    """一个模型对应一个实例。cfg.assistant_prefix 接在助手开头之后（带思考的模型是 "<think>\\n"）。"""

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
        self._url = cfg.base_url.rstrip("/") + "/api/generate"
        self._client = make_client(cfg.base_url, timeout_s, transport)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int = 0,
    ) -> Completion:
        prefix = self._cfg.assistant_prefix
        options: dict[str, float | int] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "seed": seed,
        }
        if self._cfg.num_ctx:
            options["num_ctx"] = self._cfg.num_ctx
        payload = {
            "model": self._cfg.model,
            "prompt": chatml.render(messages, tools, prefix=prefix),
            "raw": True,
            "stream": False,
            "options": options,
        }
        data, latency_ms = post_json(
            self._client,
            self._url,
            payload,
            name=self.name,
            base_url=self._cfg.base_url,
            timeout_s=self._timeout_s,
        )
        if not isinstance(data, dict) or not isinstance(data.get("response"), str):
            raise ModelError(f"模型 {self.name} 的服务返回的内容不是预期的格式：缺少字段 'response'")
        text, calls = chatml.parse(prefix + data["response"])
        # done 为 false 是服务端中途中止了生成（如同一个 token 重复太多次），这时没有结束原因与 token 数
        finish = str(data.get("done_reason") or "stop") if data.get("done") else "aborted"
        return Completion(
            text=text,
            tool_calls=calls,
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
            cached=False,
            finish_reason=finish,
        )
