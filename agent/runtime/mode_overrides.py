from __future__ import annotations

import os


def enable_a_board_quality_mode() -> None:
    os.environ["AFAC_LOGICRAG_ENABLED"] = "true"
    os.environ["AFAC_RETRIEVAL_STRATEGY"] = "logicrag_agent"
    os.environ["AFAC_A_BOARD_OPTION_MATRIX_ENABLED"] = "false"
    os.environ["AFAC_ENABLE_MULTI_OPTION_JUDGEMENT"] = "false"
    os.environ["AFAC_A_BOARD_COVERAGE_GATE_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY"] = "false"
    os.environ["AFAC_A_BOARD_MULTI_LOGICRAG_ENABLED"] = "false"
    os.environ["AFAC_A_BOARD_CLAIM_CENTRIC_MULTI_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_CLAIM_CENTRIC_MCQ_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_EVIDENCE_SET_SELECTION_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_CLAIM_SET_VERIFICATION_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_NUMERIC_FACT_LEDGER_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_CLAIM_REQUIRE_VALID_CITATIONS"] = "true"
