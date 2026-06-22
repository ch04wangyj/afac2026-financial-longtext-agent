from __future__ import annotations

from types import SimpleNamespace

from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.exhaustive_verifier import ExhaustiveVerifier, ExhaustiveVerifierConfig
from agent.schemas import Chunk, Question, TokenUsage


class _FakeLLM:
    def __init__(self, answer: str = "AC") -> None:
        self.answer = answer
        self.settings = SimpleNamespace(qwen_model="fake-qwen")

    def chat(self, messages, **kwargs):
        return LLMResponse(
            text=f'{{"answer":"{self.answer}","confidence":0.8}}',
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
        )


def _chunk(chunk_id: str, doc_id: str, text: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_reports",
        page=page,
        section="财务数据",
        clause_id="",
        text=text,
        numbers=[],
        dates=[],
        metadata={"title": doc_id},
    )


def _question() -> Question:
    return Question(
        qid="fin_test",
        domain="financial_reports",
        split="a",
        question="根据两家公司报告，哪些说法正确？",
        options={
            "A": "甲公司2025年营业收入为100亿元",
            "B": "乙公司2025年营业收入为100亿元",
            "C": "甲公司营业收入高于乙公司",
            "D": "两家公司营业收入均下降",
        },
        answer_format="multi",
        doc_ids=["doc_a", "doc_b"],
    )


def test_collect_evidence_balances_docs_and_finds_exact_values():
    index = BM25SearchIndex.build(
        [
            _chunk("a1", "doc_a", "甲公司2025年营业收入100亿元，同比增长20%。", 1),
            _chunk("a2", "doc_a", "其他说明。", 2),
            _chunk("b1", "doc_b", "乙公司2025年营业收入80亿元，同比增长10%。", 1),
            _chunk("b2", "doc_b", "其他说明。", 2),
        ]
    )
    verifier = ExhaustiveVerifier(
        index,
        _FakeLLM(),
        ExhaustiveVerifierConfig(max_context_chars=10_000, exact_sweep_per_doc=4),
    )

    evidence, report = verifier.collect_evidence(_question())

    assert {item.doc_id for item in evidence} == {"doc_a", "doc_b"}
    assert any("100亿元" in item.evidence_text for item in evidence)
    assert any("80亿元" in item.evidence_text for item in evidence)
    assert report["selected_chars"] <= 10_000


def test_solve_uses_strict_model_answer_and_tracks_usage():
    index = BM25SearchIndex.build(
        [
            _chunk("a1", "doc_a", "甲公司2025年营业收入100亿元。", 1),
            _chunk("b1", "doc_b", "乙公司2025年营业收入80亿元。", 1),
        ]
    )
    result = ExhaustiveVerifier(index, _FakeLLM("AC")).solve(_question())

    assert result.answer == "AC"
    assert result.token_usage.total_tokens == 12
    assert result.metadata["strategy"] == "exhaustive_document_verifier"
    assert result.metadata["evidence_id_map"]


def test_audit_usage_is_aggregated_and_can_replace_first_answer():
    class _AuditLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            answer = "A" if self.calls == 1 else "AC"
            return LLMResponse(
                text=f'{{"answer":"{answer}","confidence":0.9}}',
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
            )

    index = BM25SearchIndex.build(
        [
            _chunk("a1", "doc_a", "甲公司2025年营业收入100亿元。", 1),
            _chunk("b1", "doc_b", "乙公司2025年营业收入80亿元。", 1),
        ]
    )
    result = ExhaustiveVerifier(
        index,
        _AuditLLM(),
        ExhaustiveVerifierConfig(audit_enabled=True),
    ).solve(_question())

    assert result.answer == "AC"
    assert result.token_usage.total_tokens == 24
    assert result.metadata["audit_enabled"] is True
