from types import SimpleNamespace

from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.precise_verifier import (
    PreciseVerifier,
    PreciseVerifierConfig,
    _answer_from_checks,
    _format_grouped_context,
)
from agent.retrieve.claims import build_claim_targets
from agent.schemas import Chunk, Question, RetrievalResult, TokenUsage


class _FakeLLM:
    def __init__(self, answers=("AC",)) -> None:
        self.answers = iter(answers)
        self.settings = SimpleNamespace(qwen_model="fake-qwen")

    def chat(self, messages, **kwargs):
        answer = next(self.answers)
        return LLMResponse(
            text=f'{{"answer":"{answer}","confidence":0.8}}',
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
        )


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_reports",
        page=1,
        section="主要财务数据",
        clause_id="",
        text=text,
        numbers=[],
        dates=[],
        metadata={"title": doc_id, "chunk_type": "atomic_text", "hierarchy_level": "child"},
    )


def _question() -> Question:
    return Question(
        qid="fin_test",
        domain="financial_reports",
        split="a",
        question="根据两家公司报告，哪些营业收入说法正确？",
        options={
            "A": "甲公司2025年营业收入为100亿元",
            "B": "乙公司2025年营业收入为100亿元",
            "C": "甲公司营业收入高于乙公司",
        },
        answer_format="multi",
        doc_ids=["doc_a", "doc_b"],
    )


def test_precise_verifier_balances_actual_values_with_small_context():
    index = BM25SearchIndex.build(
        [
            _chunk("a", "doc_a", "甲公司2025年营业收入为100亿元，同比增长20%。"),
            _chunk("b", "doc_b", "乙公司2025年营业收入为80亿元，同比增长10%。"),
            _chunk("noise", "doc_b", "乙公司有息负债为100亿元。"),
        ]
    )
    verifier = PreciseVerifier(
        index,
        _FakeLLM(),
        PreciseVerifierConfig(max_context_chars=3000, evidence_per_claim=3),
    )

    evidence, report = verifier.collect_evidence(_question())

    assert {item.doc_id for item in evidence} == {"doc_a", "doc_b"}
    assert any("80亿元" in item.evidence_text for item in evidence)
    assert report["selected_chars"] <= 3000
    assert report["selected_count"] <= 9


def test_precise_verifier_tracks_one_pass_and_audit_usage():
    index = BM25SearchIndex.build(
        [
            _chunk("a", "doc_a", "甲公司2025年营业收入为100亿元。"),
            _chunk("b", "doc_b", "乙公司2025年营业收入为80亿元。"),
        ]
    )
    result = PreciseVerifier(
        index,
        _FakeLLM(("A", "AC")),
        PreciseVerifierConfig(audit_enabled=True),
    ).solve(_question())

    assert result.answer == "AC"
    assert result.token_usage.total_tokens == 24
    assert result.metadata["strategy"] == "v3_atomic_precise"
    assert result.metadata["evidence_id_map"]


def test_precise_verifier_can_label_layout_candidate_strategy():
    index = BM25SearchIndex.build(
        [
            _chunk("a", "doc_a", "甲公司2025年营业收入为100亿元。"),
            _chunk("b", "doc_b", "乙公司2025年营业收入为80亿元。"),
        ]
    )

    result = PreciseVerifier(
        index,
        _FakeLLM(("AC",)),
        PreciseVerifierConfig(strategy_name="v4_layout_precise"),
    ).solve(_question())

    assert result.metadata["strategy"] == "v4_layout_precise"
    assert all(item.source.startswith("v4_layout_precise:") for item in result.evidence)


def test_precise_verifier_structure_navigation_is_additive():
    chunks = [
        _chunk("p1", "doc_a", "公司概况和历史沿革。"),
        _chunk("p2", "doc_a", "营业收入为100亿元。"),
        _chunk("p3", "doc_b", "营业收入为80亿元。"),
    ]
    chunks[0].page = 1
    chunks[1].page = 2
    chunks[2].page = 5
    verifier = PreciseVerifier(
        BM25SearchIndex.build(chunks),
        _FakeLLM(),
        PreciseVerifierConfig(
            enable_structure_navigation=True,
            navigation_nodes_per_doc=1,
            navigation_candidates_per_doc=3,
        ),
    )
    claim = build_claim_targets(_question())[0]

    candidates = verifier._collect_claim_candidates(
        _question(),
        claim,
        ["营业收入"],
    )

    assert any("structure_navigation" in item.source for item in candidates)
    assert any(item.source.endswith(":mixed") for item in candidates)


def test_grouped_context_exposes_business_document_label():
    question = Question(
        qid="ins-test",
        domain="insurance",
        split="a",
        question="施救费用上限是什么？",
        options={"A": "平安特种车险最高不超过保险金额"},
        answer_format="mcq",
        doc_ids=["9"],
    )
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="9",
            domain="insurance",
            score=1.0,
            source="test",
            query="施救费用",
            evidence_text="施救费用最高不超过保险金额。",
            metadata={"option_key": "A", "title": "9", "page": 1},
        )
    ]

    context, _ = _format_grouped_context(question, evidence, 2_000)

    assert "title=平安特种车险" in context


def test_answer_from_checks_uses_selected_truth_not_freeform_answer():
    question = _question()
    response = (
        '{"checks":{"A":{"truth":"true"},"B":{"truth":"false"},'
        '"C":{"truth":"true"}},"answer":"B"}'
    )

    assert _answer_from_checks(response, question) == "AC"
