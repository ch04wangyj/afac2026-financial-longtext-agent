"""数值事实账本测试。"""

from agent.reasoning.fact_ledger import compile_numeric_fact_ledger, format_numeric_fact_ledger
from agent.schemas import Question, RetrievalResult


def _question(domain: str = "financial_reports") -> Question:
    return Question(
        qid="q1",
        domain=domain,
        split="A",
        question="2025年营业收入是否高于2024年？",
        options={"A": "2025年营业收入高于2024年"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )


def _evidence(text: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            doc_id="doc1",
            domain="financial_reports",
            score=8.0,
            source="test",
            query="营业收入",
            evidence_text=text,
            metadata={"chunk_type": "table"},
        )
    ]


def test_fact_ledger_normalizes_header_unit_and_keeps_sources():
    ledger = compile_numeric_fact_ledger(
        _question(),
        _evidence("单位：千元 2025年营业收入 456,451,731；2024年营业收入 409,084,000。"),
    )

    values = {fact["raw_value"]: fact for fact in ledger["facts"]}
    assert values["456,451,731"]["normalized_value"] == "456451731000"
    assert values["456,451,731"]["metric"] == "营业收入"
    assert values["456,451,731"]["doc_id"] == "doc1"
    assert values["456,451,731"]["chunk_id"] == "c1"


def test_fact_ledger_treats_parentheses_as_negative_and_percent_as_ratio():
    ledger = compile_numeric_fact_ledger(
        _question(),
        _evidence("净利润（1,234）万元，资产负债率为63.5%。"),
    )

    normalized = {fact["raw_value"]: fact["normalized_value"] for fact in ledger["facts"]}
    assert normalized["(1,234)"] == "-12340000"
    assert normalized["63.5"] == "0.635"


def test_fact_ledger_skips_plain_years_but_formats_real_values():
    ledger = compile_numeric_fact_ledger(_question(), _evidence("2025年营业收入为120亿元。"))

    assert all(fact["raw_value"] != "2025" for fact in ledger["facts"])
    rendered = format_numeric_fact_ledger(ledger)
    assert "normalized=12000000000" in rendered
    assert "doc=doc1" in rendered
