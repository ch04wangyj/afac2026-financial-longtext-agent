from types import SimpleNamespace

from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.precise_verifier import (
    PreciseVerifier,
    PreciseVerifierConfig,
    _answer_from_checks,
    _format_grouped_context,
    _format_numeric_verification_report,
    build_tf_judge_messages,
    build_precise_judge_messages,
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


def test_answer_from_checks_uses_fact_and_question_applicability_layers():
    question = _question()
    response = (
        '{"checks":{'
        '"A":{"fact_truth":"true","applicable":"true","selected":"true"},'
        '"B":{"fact_truth":"true","applicable":"false","selected":"true"},'
        '"C":{"fact_truth":"false","applicable":"true","selected":"false"}'
        '},"answer":"AB"}'
    )

    assert _answer_from_checks(response, question) == "A"


def test_numeric_verification_context_exposes_normalized_operands():
    report = {
        "facts": [
            {
                "fact_id": "F1",
                "doc_id": "a",
                "metric": "营业收入",
                "year": "2024",
                "raw_value": "777",
                "unit": "亿元",
                "normalized_value": "77700000000",
                "scope": "consolidated",
            },
            {
                "fact_id": "F2",
                "doc_id": "b",
                "metric": "营业收入",
                "year": "2024",
                "raw_value": "407",
                "unit": "亿元",
                "normalized_value": "40700000000",
                "scope": "consolidated",
            },
        ],
        "calculations": [
            {
                "operation": "compare",
                "expression": "F1 compare F2 = gt",
                "result": "gt",
            }
        ],
    }

    context = _format_numeric_verification_report(report)

    assert "normalized=77700000000" in context
    assert "F1 compare F2 = gt" in context


def test_tf_prompt_maps_proposition_truth_to_a_or_b_once():
    question = Question(
        qid="tf_test",
        domain="regulatory",
        split="a",
        question="该规定自2026年1月1日起施行。",
        options={"A": "正确", "B": "错误"},
        answer_format="tf",
        doc_ids=["doc1"],
    )

    messages = build_tf_judge_messages(question, "2026年1月1日起施行。")
    text = "\n".join(message["content"] for message in messages)

    assert "命题全部成立" in text
    assert "answer=A" in text
    assert "任一子句不成立" in text
    assert "answer=B" in text


def test_question_envelope_only_activates_scope_split_for_explicit_set_query():
    scope_question = Question(
        qid="scope_test",
        domain="regulatory",
        split="a",
        question="下列哪些情形需要内部审批或满足金额门槛？",
        options={"A": "需要董事会审批", "B": "记录保存五年"},
        answer_format="multi",
        doc_ids=["doc1"],
    )
    scope_text = "\n".join(
        message["content"]
        for message in build_precise_judge_messages(
            scope_question,
            "原文",
            enable_question_envelope=True,
        )
    )
    generic_text = "\n".join(
        message["content"]
        for message in build_precise_judge_messages(
            _question(),
            "原文",
            enable_question_envelope=True,
        )
    )

    assert '"fact_truth":"true|false|uncertain"' in scope_text
    assert '"applicable":"true|false|uncertain"' in scope_text
    assert '"truth":"true|false|uncertain"' in generic_text
