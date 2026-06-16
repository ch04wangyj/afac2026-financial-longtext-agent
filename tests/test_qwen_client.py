from __future__ import annotations

from dataclasses import replace

import pytest

from agent.config import Settings
from agent.llm import qwen_client as qwen_client_module
from agent.llm.qwen_client import QwenClient


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}: {self.text}")

    def json(self) -> dict:
        return self._payload


def _settings() -> Settings:
    return replace(
        Settings.from_env(),
        qwen_model="qwen-plus",
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        qwen_enable_thinking=False,
        request_timeout_seconds=30,
        max_retries=0,
    )


def test_qwen_client_non_stream_request_uses_dashscope_http_json(monkeypatch):
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
            }
        )

    monkeypatch.setattr(qwen_client_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(qwen_client_module.requests, "post", fake_post)

    client = QwenClient(_settings(), dry_run=False)
    response = client.chat([{"role": "user", "content": "只回复OK"}], max_tokens=16)

    assert response.text == "OK"
    assert response.usage.total_tokens == 12
    assert response.reasoning == ""
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "qwen-plus"
    assert captured["json"]["enable_thinking"] is False
    assert "extra_body" not in captured["json"]
    assert captured["timeout"] == 30


def test_qwen_client_falls_back_to_estimated_usage_when_usage_missing(monkeypatch):
    monkeypatch.setattr(qwen_client_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        qwen_client_module.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"choices": [{"message": {"content": "ABCDE"}}]}),
    )

    client = QwenClient(_settings(), dry_run=False)
    response = client.chat([{"role": "user", "content": "1234567890"}], max_tokens=16)

    assert response.text == "ABCDE"
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 7


def test_qwen_client_retries_then_raises_runtimeerror_with_context(monkeypatch):
    attempts = {"n": 0}

    def fake_post(*args, **kwargs):
        attempts["n"] += 1
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(qwen_client_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(qwen_client_module.requests, "post", fake_post)

    client = QwenClient(replace(_settings(), max_retries=1), dry_run=False)

    with pytest.raises(RuntimeError, match="Qwen API call failed after retries") as exc:
        client.chat([{"role": "user", "content": "只回复OK"}], max_tokens=16)

    assert attempts["n"] == 2
    assert "provider timed out" in str(exc.value)


def test_qwen_client_dry_run_still_returns_dummy_response_without_http(monkeypatch):
    called = {"post": 0}

    def fake_post(*args, **kwargs):
        called["post"] += 1
        raise AssertionError("HTTP should not be called during dry-run")

    monkeypatch.setattr(qwen_client_module.requests, "post", fake_post)

    client = QwenClient(_settings(), dry_run=True)
    response = client.chat([{"role": "user", "content": "hello world"}], max_tokens=16)

    assert response.text == '{"answer":"A","confidence":0.0,"reason":"dry-run"}'
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens == 1
    assert called["post"] == 0
