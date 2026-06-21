from __future__ import annotations

import os

import pytest

from agent.config import Settings
from agent.retrieve.retriever import Retriever
from agent.runtime.strategy_contract import validate_runtime_strategy


class DummyIndex:
    def __init__(self) -> None:
        self.default_search_mode = None


class DummyDocIndex:
    def search_doc_ids(self, query: str, top_k: int, domain: str | None = None) -> list[str]:
        return []


def test_runtime_strategy_accepts_only_mainline_and_logicrag():
    assert validate_runtime_strategy("doc_first_bm25f_expansion") == "doc_first_bm25f_expansion"
    assert validate_runtime_strategy("logicrag_agent") == "logicrag_agent"
    assert validate_runtime_strategy("logicrag_qwen_rrf") == "logicrag_qwen_rrf"

    for legacy in [
        "hybrid",
        "logicrag",
        "question_options",
        "rule_multi_rrf",
        "field_boosted_rrf",
        "logic_lite_rrf",
        "linear_entity_rrf",
        "graph_lite_rrf",
        "crag_lite",
    ]:
        with pytest.raises(ValueError):
            validate_runtime_strategy(legacy)


def test_retriever_rejects_hybrid_alias():
    with pytest.raises(ValueError):
        Retriever(DummyIndex(), strategy="hybrid")


def test_retriever_rejects_logicrag_alias():
    with pytest.raises(ValueError):
        Retriever(DummyIndex(), strategy="logicrag")


def test_settings_from_env_rejects_removed_runtime_alias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFAC_RETRIEVAL_STRATEGY", "hybrid")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_enable_a_board_quality_mode_sets_logicrag_runtime_env(monkeypatch: pytest.MonkeyPatch):
    from agent.runtime.mode_overrides import enable_a_board_quality_mode

    for key in [
        "AFAC_LOGICRAG_ENABLED",
        "AFAC_RETRIEVAL_STRATEGY",
        "AFAC_A_BOARD_OPTION_MATRIX_ENABLED",
        "AFAC_ENABLE_MULTI_OPTION_JUDGEMENT",
        "AFAC_A_BOARD_COVERAGE_GATE_ENABLED",
        "AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED",
        "AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY",
    ]:
        monkeypatch.delenv(key, raising=False)

    enable_a_board_quality_mode()

    assert os.environ["AFAC_LOGICRAG_ENABLED"] == "true"
    assert os.environ["AFAC_RETRIEVAL_STRATEGY"] == "logicrag_agent"
    assert os.environ["AFAC_A_BOARD_OPTION_MATRIX_ENABLED"] == "false"
    assert os.environ["AFAC_ENABLE_MULTI_OPTION_JUDGEMENT"] == "false"
    assert os.environ["AFAC_A_BOARD_COVERAGE_GATE_ENABLED"] == "true"
    assert os.environ["AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED"] == "true"
    assert os.environ["AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY"] == "false"
