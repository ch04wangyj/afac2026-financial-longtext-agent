import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "13_augment_financial_metric_rows.py"
SPEC = importlib.util.spec_from_file_location("augment_financial_rows", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_augmenter_keeps_base_rows_and_adds_structured_financial_rows():
    rows = [
        {
            "chunk_id": "parent",
            "doc_id": "annual_a",
            "domain": "financial_reports",
            "page": 3,
            "section": "主要财务数据",
            "clause_id": "",
            "text": "2025 年 2024 年\n营业收入(亿元) 120 100",
            "tables": [],
            "numbers": ["2025", "2024", "120", "100"],
            "dates": [],
            "metadata": {"title": "甲公司年报", "chunk_type": "text"},
        }
    ]

    augmented, count = MODULE.augment_financial_metric_rows(rows)

    assert count == 1
    assert len(augmented) == 2
    generated = augmented[1]
    assert generated["metadata"]["chunk_type"] == "financial_metric_row"
    assert generated["metadata"]["parent_chunk_id"] == "parent"
    assert generated["metadata"]["financial_row"]["metric"] == "营业收入"
