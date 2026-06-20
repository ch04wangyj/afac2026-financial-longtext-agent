from __future__ import annotations

from pathlib import Path

import pytest

from agent.config import Settings
from agent.index.bm25 import BM25SearchIndex


from agent.retrieve.probe_cases import PROBE_CASES, PROBE_CASES_BY_NAME

TARGET_DOC_ID = "annual_byd_2025_report"
TARGET_RAW_PATH = Path(
    "public_dataset_a/public_dataset_upload/raw/financial_reports/annual_byd_2025_report.PDF"
)
TARGET_ANSWER_TERMS = PROBE_CASES_BY_NAME["byd_2025_net_profit"]["target_answer_terms"]
KEYWORD_BUNDLES = PROBE_CASES_BY_NAME["byd_2025_net_profit"]["keyword_bundles"]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def bm25_index(settings: Settings) -> BM25SearchIndex:
    index_path = settings.index_dir / "bm25_index.pkl"
    if not index_path.exists():
        pytest.skip(f"missing retrieval index: {index_path}")
    return BM25SearchIndex.load(index_path)
