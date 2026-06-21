"""claim 集合校准测试。"""

from agent.reasoning.claim_set_verifier import (
    aggregate_claim_relations,
    calibrate_claim_verdict,
    should_run_claim_set_verification,
    validate_evidence_refs,
)
from agent.schemas import Question, RetrievalResult


def _evidence(doc_id: str, chunk_id: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_reports",
        score=1.0,
        source="test",
        query="",
        evidence_text="2025年营业收入为120亿元。",
    )


def _question(answer_format: str = "multi") -> Question:
    return Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="以下说法正确的是？",
        options={"A": "两家公司营业收入均增长", "B": "净利润下降"},
        answer_format=answer_format,
        doc_ids=["doc1", "doc2"],
    )


def test_validate_evidence_refs_drops_invalid_and_duplicate_ids():
    assert validate_evidence_refs(["[1]", "证据1", "[9]", "x"], evidence_count=2) == [1]


def test_calibration_downgrades_support_without_valid_citation():
    calibrated = calibrate_claim_verdict(
        relation="support",
        confidence=0.95,
        support_evidence=["[9]"],
        refute_evidence=[],
        sufficiency={"sufficient": True},
        selection_report={"missing_slots": []},
        evidence=[_evidence("doc1", "c1")],
        doc_scope=["doc1"],
        option_text="营业收入增长",
    )

    assert calibrated["relation"] == "insufficient"
    assert "missing_relation_citation" in calibrated["calibration_tags"]


def test_calibration_requires_all_docs_for_universal_claim_support():
    calibrated = calibrate_claim_verdict(
        relation="support",
        confidence=0.9,
        support_evidence=["[1]"],
        refute_evidence=[],
        sufficiency={"sufficient": True},
        selection_report={"missing_slots": []},
        evidence=[_evidence("doc1", "c1"), _evidence("doc2", "c2")],
        doc_scope=["doc1", "doc2"],
        option_text="两家公司营业收入均增长",
    )

    assert calibrated["relation"] == "insufficient"
    assert calibrated["missing_universal_doc_ids"] == ["doc2"]


def test_calibration_keeps_universal_support_when_citations_cover_all_docs():
    calibrated = calibrate_claim_verdict(
        relation="support",
        confidence=0.9,
        support_evidence=["[1]", "[2]"],
        refute_evidence=[],
        sufficiency={"sufficient": True},
        selection_report={"missing_slots": []},
        evidence=[_evidence("doc1", "c1"), _evidence("doc2", "c2")],
        doc_scope=["doc1", "doc2"],
        option_text="两家公司营业收入均增长",
    )

    assert calibrated["relation"] == "support"


def test_aggregate_and_trigger_set_verification_for_exact_match_multi():
    question = _question()
    answer, report = aggregate_claim_relations(
        question,
        {
            "A": {"relation": "support", "confidence": 0.8},
            "B": {"relation": "refute", "confidence": 0.9},
        },
    )

    assert answer == "A"
    assert report["supported_options"] == ["A"]
    assert should_run_claim_set_verification(question, {}) is True
