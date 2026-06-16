"""百炼兼容 Qwen 客户端封装。

所有真实模型调用都经过这里，统一走稳定的 DashScope HTTP JSON 路径，
避免 SDK streaming 路径在 qwen-plus 上卡死/超时。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

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
        """调用 Qwen 聊天接口；真实调用固定走非流式 HTTP JSON 路径。"""
        if self.dry_run:
            prompt_chars = sum(len(message.get("content", "")) for message in messages)
            usage = TokenUsage(prompt_tokens=max(1, prompt_chars // 2), completion_tokens=1)
            return LLMResponse(text='{"answer":"A","confidence":0.0,"reason":"dry-run"}', usage=usage)

        last_error: Exception | None = None
        thinking = self.settings.qwen_enable_thinking if enable_thinking is None else enable_thinking
        url = self.settings.qwen_base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self.settings.max_retries + 1):
            try:
                payload = {
                    "model": self.settings.qwen_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "enable_thinking": thinking,
                }
                response = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.settings.request_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                message = ((data.get("choices") or [{}])[0].get("message") or {})
                text = message.get("content", "") or ""
                reasoning = message.get("reasoning_content", "") or ""
                usage = self._usage_from_dashscope_json(data, messages, text)
                return LLMResponse(text=text, usage=usage, reasoning=reasoning, raw=data)
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Qwen API call failed after retries: {last_error}") from last_error

    @staticmethod
    def _usage_from_dashscope_json(data: dict, messages: list[dict[str, str]], output_text: str) -> TokenUsage:
        """从 DashScope JSON 响应中提取 usage；若缺失则退回估算。"""
        raw_usage = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
        )
        if usage.total_tokens == 0:
            estimated = QwenClient._estimate_usage(messages, output_text)
            usage = TokenUsage(
                prompt_tokens=estimated.prompt_tokens,
                completion_tokens=estimated.completion_tokens,
                total_tokens=estimated.total_tokens,
            )
        return usage

    @staticmethod
    def _estimate_usage(messages: list[dict[str, str]], output_text: str) -> TokenUsage:
        """当接口未返回 usage 时，用字符数做保守估算。"""
        prompt_chars = sum(len(message.get("content", "")) for message in messages)
        completion_chars = len(output_text)
        return TokenUsage(
            prompt_tokens=max(1, prompt_chars // 2),
            completion_tokens=max(1, completion_chars // 2),
        )
