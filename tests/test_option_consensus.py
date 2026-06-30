from agent.evaluation.option_consensus import (
    audit_option_consensus,
    candidate_answer_from_consensus,
)
from agent.schemas import AnswerResult, Question, TokenUsage


def _question() -> Question:
    return Question(
        qid="q1",
        domain="insurance",
        split="a",
        question="哪些说法正确？",
        options={"A": "直接条款", "B": "缺失声明", "C": "其他"},
        answer_format="multi",
        doc_ids=["d1"],
    )


def _result(answer: str) -> AnswerResult:
    return AnswerResult(
        qid="q1",
        answer=answer,
        confidence=1.0,
        evidence=[],
        token_usage=TokenUsage(),
    )


def test_option_consensus_requires_independent_unanimous_flip():
    rows = audit_option_consensus(
        [_question()],
        [_result("AB")],
        {
            "precise": [_result("A")],
            "exhaustive": [_result("AC")],
        },
        min_runs=2,
    )
    by_key = {row.option_key: row for row in rows}

    assert by_key["B"].unanimous_flip is True
    assert by_key["C"].unanimous_flip is False
    assert candidate_answer_from_consensus(_question(), "AB", rows) == "A"


def test_option_consensus_accepts_partial_runs_without_counting_missing_vote():
    other = Question(
        qid="q2",
        domain="insurance",
        split="a",
        question="判断。",
        options={"A": "正确", "B": "错误"},
        answer_format="tf",
        doc_ids=[],
    )
    rows = audit_option_consensus(
        [_question(), other],
        [_result("AB")],
        {"partial": []},
        min_runs=1,
    )

    assert all(row.observed_runs == 0 for row in rows)
