from agent.reasoning.question_envelope import (
    build_question_envelope,
    selection_verdict,
)
from agent.schemas import Question


def _question(text: str, *, doc_ids: list[str] | None = None) -> Question:
    return Question(
        qid="q1",
        domain="regulatory",
        split="a",
        question=text,
        options={"A": "选项一", "B": "选项二"},
        answer_format="multi",
        doc_ids=doc_ids or ["d1"],
    )


def test_question_envelope_detects_incorrect_selection_and_scope():
    envelope = build_question_envelope(
        _question("下列哪些情形需要内部审批或满足金额门槛？")
    )

    assert envelope.selection_rule == "correct"
    assert envelope.scope_markers == ("审批", "金额", "门槛", "情形")
    assert envelope.scope_gate_enabled is True
    assert "内部审批" in envelope.focus
    assert "金额门槛" in envelope.focus


def test_question_envelope_marks_cross_document_comparison():
    envelope = build_question_envelope(
        _question(
            "两份报告的营业收入均高于上年，以下说法正确的是？",
            doc_ids=["d1", "d2"],
        )
    )

    assert envelope.requires_all_documents is True
    assert envelope.requires_comparison is True
    assert envelope.scope_gate_enabled is False


def test_selection_verdict_separates_fact_from_question_applicability():
    assert selection_verdict("true", "false", "correct") == "false"
    assert selection_verdict("true", "true", "correct") == "true"
    assert selection_verdict("false", "true", "incorrect") == "true"
    assert selection_verdict("uncertain", "true", "correct") == "uncertain"
