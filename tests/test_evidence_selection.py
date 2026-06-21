"""slot-aware 证据集合选择测试。"""

from agent.retrieve.evidence_selection import select_evidence_set
from agent.retrieve.targets import RetrievalTarget
from agent.schemas import RetrievalResult


def _result(chunk_id: str, doc_id: str, text: str, score: float, **metadata) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_reports",
        score=score,
        source="test",
        query="营业收入",
        evidence_text=text,
        metadata=metadata,
    )


def test_selector_prefers_doc_and_comparison_endpoint_coverage_over_duplicates():
    target = RetrievalTarget(
        node_id="q:A",
        rank=0,
        question="两家公司2025年营业收入是否均高于2024年",
        doc_scope=["doc_a", "doc_b"],
        must_terms=["营业收入", "2025", "2024"],
        numbers=["2025", "2024"],
        dates=["2025年", "2024年"],
        evidence_intent="comparison",
    )
    candidates = [
        _result("a1", "doc_a", "2025年营业收入为120亿元，2024年为100亿元。", 10.0, chunk_type="table"),
        _result("a2", "doc_a", "2025年营业收入为120亿元，2024年为100亿元。", 9.9, chunk_type="table"),
        _result("b1", "doc_b", "2025年营业收入为80亿元，2024年为90亿元。", 6.0, chunk_type="table"),
    ]

    selected, report = select_evidence_set(target, candidates, top_k=2, max_chars=1000)

    assert [item.chunk_id for item in selected] == ["a1", "b1"]
    assert report.missing_slots == []
    assert report.coverage_ratio == 1.0


def test_selector_reports_missing_numeric_value_when_only_years_are_present():
    target = RetrievalTarget(
        node_id="q:A",
        rank=0,
        question="比较2025年与2024年营业收入",
        doc_scope=["doc_a"],
        must_terms=["营业收入"],
        numbers=["2025", "2024"],
        dates=[],
        evidence_intent="comparison",
    )
    candidates = [_result("a1", "doc_a", "本节讨论2025年与2024年营业收入。", 10.0)]

    _, report = select_evidence_set(target, candidates, top_k=2, max_chars=1000)

    assert "fact:numeric_value" in report.missing_slots
    assert "fact:comparison_endpoints" in report.missing_slots


def test_selector_respects_character_budget_after_first_evidence():
    target = RetrievalTarget(node_id="q", rank=0, question="事实核对", doc_scope=[], evidence_intent="fact")
    candidates = [
        _result("a", "doc", "甲" * 80, 2.0),
        _result("b", "doc", "乙" * 80, 1.0),
    ]

    selected, report = select_evidence_set(target, candidates, top_k=2, max_chars=100)

    assert len(selected) == 1
    assert report.used_chars == 80


def test_selector_keeps_pinned_verdict_evidence_before_higher_scored_duplicates():
    target = RetrievalTarget(node_id="q", rank=0, question="核对结论", doc_scope=["doc_a"], evidence_intent="fact")
    candidates = [
        _result("high", "doc_a", "高分但未被 verdict 引用。", 10.0),
        _result("cited", "doc_a", "局部 verdict 直接引用的原文。", 1.0),
    ]

    selected, _ = select_evidence_set(
        target,
        candidates,
        top_k=1,
        max_chars=1000,
        pinned_chunk_ids={"cited"},
    )

    assert [item.chunk_id for item in selected] == ["cited"]


def test_selector_rewards_exact_metric_alias_evidence():
    target = RetrievalTarget(node_id="q", rank=0, question="比较净利润", doc_scope=["doc"], evidence_intent="comparison")
    generic = _result("generic", "doc", "2025年公司年度报告经营情况。", 10.0)
    metric = _result("metric", "doc", "归属于上市公司股东的净利润同比下降18.97%。", 5.0)
    metric.metadata["claim_metric_alias_hits"] = ["归属于上市公司股东的净利润", "本年比上年增减"]

    selected, _ = select_evidence_set(target, [generic, metric], top_k=1, max_chars=1000)

    assert [item.chunk_id for item in selected] == ["metric"]
