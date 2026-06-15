"""百炼 OpenAI-compatible Qwen 客户端封装。

所有真实模型调用都经过这里，便于统一开关 thinking、streaming、重试和 Token 统计。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from agent.config import Settings, get_api_key
from agent.schemas import TokenUsage


@dataclass
class LLMResponse:
    """模型响应的统一包装，包含回答正文、思考内容和 Token 用量。"""

    text: str
    usage: TokenUsage
    reasoning: str = ""
    raw: dict | None = None


class QwenClient:
    """Qwen API 客户端；dry_run 模式用于不消耗 Token 的链路测试。"""

    def __init__(self, settings: Settings, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self.api_key = get_api_key()
        if not dry_run and not self.api_key:
            raise RuntimeError(
                "Missing Qwen API key. Set DASHSCOPE_API_KEY/BAILIAN_API_KEY/QWEN_API_KEY "
                "or run scripts with --dry-run."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_thinking: bool | None = None,
    ) -> LLMResponse:
        """调用 Qwen 聊天接口；真实调用时走 streaming 以兼容 reasoning_content。"""
        if self.dry_run:
            prompt_chars = sum(len(message.get("content", "")) for message in messages)
            usage = TokenUsage(prompt_tokens=max(1, prompt_chars // 2), completion_tokens=1)
            return LLMResponse(text='{"answer":"A","confidence":0.0,"reason":"dry-run"}', usage=usage)

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.settings.qwen_base_url)
        last_error: Exception | None = None
        thinking = self.settings.qwen_enable_thinking if enable_thinking is None else enable_thinking
        for attempt in range(self.settings.max_retries + 1):
            try:
                # DashScope 兼容 OpenAI SDK，但 thinking 需要通过 extra_body 传递。
                kwargs = {
                    "model": self.settings.qwen_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": self.settings.request_timeout_seconds,
                    "extra_body": {"enable_thinking": thinking},
                }
                if self.settings.qwen_stream:
                    if self.settings.qwen_stream_include_usage:
                        kwargs["stream_options"] = {"include_usage": True}
                    response = client.chat.completions.create(**kwargs, stream=True)
                    return self._collect_stream_response(response, messages)

                response = client.chat.completions.create(**kwargs, stream=False)
                text = response.choices[0].message.content or ""
                raw_usage = response.usage
                usage = self._usage_from_openai(raw_usage)
                return LLMResponse(text=text, usage=usage, raw=response.model_dump())
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Qwen API call failed after retries: {last_error}") from last_error

    def _collect_stream_response(self, response, messages: list[dict[str, str]]) -> LLMResponse:
        """收集流式响应，同时兼容 reasoning_content 和 usage chunk。"""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        raw_chunks: list[dict] = []
        usage = TokenUsage()
        for chunk in response:
            try:
                raw_chunks.append(chunk.model_dump())
            except Exception:
                pass
            if getattr(chunk, "usage", None) is not None:
                usage = self._usage_from_openai(chunk.usage)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            content = getattr(delta, "content", None)
            if content:
                content_parts.append(content)

        if usage.total_tokens == 0:
            usage = self._estimate_usage(messages, "".join(content_parts))
        return LLMResponse(
            text="".join(content_parts),
            reasoning="".join(reasoning_parts),
            usage=usage,
            raw={"chunks": raw_chunks},
        )

    @staticmethod
    def _usage_from_openai(raw_usage) -> TokenUsage:
        """把 OpenAI SDK 的 usage 对象转换成本项目 schema。"""
        return TokenUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
        )

    @staticmethod
    def _estimate_usage(messages: list[dict[str, str]], output_text: str) -> TokenUsage:
        """当流式接口未返回 usage 时，用字符数做保守估算。"""
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        completion_chars = len(output_text)
        return TokenUsage(
            prompt_tokens=max(1, prompt_chars // 2),
            completion_tokens=max(1, completion_chars // 2),
        )
