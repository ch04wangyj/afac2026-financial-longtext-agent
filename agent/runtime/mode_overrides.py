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
