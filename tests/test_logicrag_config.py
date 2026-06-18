from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime.logicrag_config import load_logicrag_runtime_config


def test_load_logicrag_runtime_config_reads_profiles_from_yaml():
    config = load_logicrag_runtime_config(
        Path("D:/pyproject/afac2026-financial-longtext-agent/configs/logicrag_runtime.yaml")
    )

    assert config.qwen.model == "qwen3.7-plus"
    assert config.concurrency.question_workers == 15
    assert config.concurrency.qwen_workers == 8
    assert config.thinking_profiles["logicrag_planner"].enabled is True
    assert config.thinking_profiles["logicrag_planner"].max_tokens == 1024
    assert config.thinking_profiles["option_judgement"].enabled is True


def test_load_logicrag_runtime_config_merges_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFAC_LOGICRAG_QWEN_MODEL", "qwen3.7-max")
    monkeypatch.setenv("AFAC_LOGICRAG_QUESTION_WORKERS", "6")
    monkeypatch.setenv("AFAC_LOGICRAG_PROFILE_LOGICRAG_PLANNER_MAX_TOKENS", "2048")

    config = load_logicrag_runtime_config(
        Path("D:/pyproject/afac2026-financial-longtext-agent/configs/logicrag_runtime.yaml")
    )

    assert config.qwen.model == "qwen3.7-max"
    assert config.concurrency.question_workers == 6
    assert config.thinking_profiles["logicrag_planner"].max_tokens == 2048
