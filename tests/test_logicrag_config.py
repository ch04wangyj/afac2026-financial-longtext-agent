from __future__ import annotations

from pathlib import Path

import pytest

from agent.runtime.logicrag_config import load_logicrag_runtime_config

CONFIG_PATH = Path("D:/pyproject/afac2026-financial-longtext-agent/configs/logicrag_runtime.yaml")


def test_load_logicrag_runtime_config_reads_profiles_from_yaml():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.qwen.model == "qwen3.7-plus"
    assert config.concurrency.question_workers == 15
    assert config.concurrency.qwen_workers == 8
    assert config.thinking_profiles["logicrag_planner"].enabled is True
    assert config.thinking_profiles["logicrag_planner"].max_tokens == 1024
    assert config.thinking_profiles["option_judgement"].enabled is False


def test_runtime_config_exposes_step_specific_budget_defaults():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.logicrag.rank_top_k == 5
    assert config.thinking_profiles["logicrag_planner"].enabled is True
    assert config.thinking_profiles["logicrag_planner"].max_tokens == 1024
    assert config.thinking_profiles["logicrag_final_compose"].enabled is True
    assert config.thinking_profiles["logicrag_rank_summary"].max_tokens < config.thinking_profiles["logicrag_final_compose"].max_tokens
    assert config.thinking_profiles["option_judgement"].enabled is False
    assert config.thinking_profiles["option_judgement"].max_tokens <= 256


def test_runtime_config_exposes_explicit_budget_hierarchy_levels():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.thinking_profiles["logicrag_planner"].level == "high"
    assert config.thinking_profiles["logicrag_final_compose"].level == "high"
    assert config.thinking_profiles["logicrag_rank_summary"].level == "medium"
    assert config.thinking_profiles["multi_option_fallback"].level == "medium"
    assert config.thinking_profiles["answer_single_pass"].level == "low"
    assert config.thinking_profiles["option_judgement"].level == "low"

    high = config.thinking_profiles["logicrag_planner"].max_tokens
    medium = config.thinking_profiles["logicrag_rank_summary"].max_tokens
    low = config.thinking_profiles["option_judgement"].max_tokens
    assert high > medium > low


def test_runtime_config_exposes_paper_faithful_logicrag_contract_defaults():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.logicrag.execution_mode == "paper_faithful_core"
    assert config.logicrag.retrieval_backend == "paper_contract_embedding"
    assert config.logicrag.dynamic_dag_augmentation is True
    assert config.logicrag.append_unresolved_after_current_rank is True
    assert config.logicrag.sampling_without_replacement is True


def test_logicrag_config_loads_adaptive_retrieval_fields(tmp_path):
    config_path = tmp_path / "logicrag_runtime.yaml"
    config_path.write_text(
        """
logicrag:
  adaptive_retrieval_enabled: true
  llm_query_bundles_enabled: true
  max_query_bundles_per_rank: 4
  max_refinement_rounds_per_rank: 1
  b_board_scope_narrowing_enabled: true
  narrowed_doc_top_n: 5
thinking_profiles:
  logicrag_query_bundle:
    level: medium
    enabled: true
    max_tokens: 512
  logicrag_sufficiency_gate:
    level: low
    enabled: false
    max_tokens: 192
  logicrag_refinement:
    level: medium
    enabled: true
    max_tokens: 512
""",
        encoding="utf-8",
    )

    config = load_logicrag_runtime_config(config_path)

    assert config.logicrag.adaptive_retrieval_enabled is True
    assert config.logicrag.llm_query_bundles_enabled is True
    assert config.logicrag.max_query_bundles_per_rank == 4
    assert config.logicrag.max_refinement_rounds_per_rank == 1
    assert config.logicrag.b_board_scope_narrowing_enabled is True
    assert config.logicrag.narrowed_doc_top_n == 5
    assert config.thinking_profiles["logicrag_query_bundle"].max_tokens == 512
    assert config.thinking_profiles["logicrag_sufficiency_gate"].enabled is False
    assert config.thinking_profiles["logicrag_refinement"].enabled is True


def test_runtime_config_supports_multi_option_fallback_profile():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.thinking_profiles["multi_option_fallback"].enabled is True
    assert config.thinking_profiles["multi_option_fallback"].max_tokens == 512


def test_load_logicrag_runtime_config_merges_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFAC_LOGICRAG_QWEN_MODEL", "qwen3.7-max")
    monkeypatch.setenv("AFAC_LOGICRAG_QUESTION_WORKERS", "6")
    monkeypatch.setenv("AFAC_LOGICRAG_PROFILE_LOGICRAG_PLANNER_MAX_TOKENS", "2048")

    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.qwen.model == "qwen3.7-max"
    assert config.concurrency.question_workers == 6
    assert config.thinking_profiles["logicrag_planner"].max_tokens == 2048


def test_runtime_config_exposes_a_board_quality_flags():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.a_board.option_matrix_enabled is False
    assert config.a_board.multi_logicrag_enabled is True
    assert config.a_board.multi_logicrag_retry_enabled is True
    assert config.a_board.coverage_gate_enabled is False
    assert config.a_board.force_doc_coverage_for_a_board is True
    assert config.a_board.use_doc_ids_as_hint_only is False
    assert config.a_board.financial_calculator_enabled is False


def test_runtime_config_exposes_multi_logicrag_profiles():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.thinking_profiles["multi_logicrag_option_planner"].max_tokens <= 256
    assert config.thinking_profiles["multi_logicrag_option_verdict"].max_tokens <= 256
    assert config.thinking_profiles["multi_logicrag_option_retry"].enabled is True
    assert config.thinking_profiles["multi_logicrag_numeric_verifier"].max_tokens >= 256



def test_claim_centric_runtime_defaults_are_safe():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.a_board.claim_centric_multi_enabled is False
    assert config.a_board.claim_centric_mcq_enabled is False
    assert config.a_board.max_claim_refinement_rounds == 1



def test_claim_centric_budget_defaults_are_token_conservative():
    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.a_board.max_claim_query_bundles <= 5
    assert config.a_board.max_claim_refinement_rounds <= 1
    assert config.a_board.claim_final_compose_enabled is False
    assert config.a_board.claim_verdict_max_evidence_items == 6
    assert config.a_board.claim_retry_verdict_max_evidence_items == 8


def test_a_board_runtime_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFAC_A_BOARD_OPTION_MATRIX_ENABLED", "true")
    monkeypatch.setenv("AFAC_A_BOARD_MULTI_LOGICRAG_ENABLED", "false")
    monkeypatch.setenv("AFAC_A_BOARD_MULTI_LOGICRAG_RETRY_ENABLED", "false")
    monkeypatch.setenv("AFAC_A_BOARD_COVERAGE_GATE_ENABLED", "true")
    monkeypatch.setenv("AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED", "true")
    monkeypatch.setenv("AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY", "false")

    config = load_logicrag_runtime_config(CONFIG_PATH)

    assert config.a_board.option_matrix_enabled is True
    assert config.a_board.multi_logicrag_enabled is False
    assert config.a_board.multi_logicrag_retry_enabled is False
    assert config.a_board.coverage_gate_enabled is True
    assert config.a_board.financial_calculator_enabled is True
    assert config.a_board.use_doc_ids_as_hint_only is False
