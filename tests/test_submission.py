from __future__ import annotations

import pytest

from agent.io.submission import merge_answer_results, validate_answer_results
from agent.schemas import AnswerResult, Question, TokenUsage


def _question(qid: str, answer_format: str = "multi") -> Question:
    return Question(
        qid=qid,
        domain="financial_reports",
        split="a",
        question="测试题",
        options={"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
        answer_format=answer_format,
    )


def _result(qid: str, answer: str, total: int = 3) -> AnswerResult:
    return AnswerResult(
        qid=qid,
        answer=answer,
        confidence=1.0,
        evidence=[],
        token_usage=TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=total),
    )


def test_merge_answer_results_replaces_by_qid_and_preserves_order():
    merged = merge_answer_results(
        [_result("q1", "A"), _result("q2", "B")],
        [[_result("q2", "AC")]],
    )

    assert [result.qid for result in merged] == ["q1", "q2"]
    assert [result.answer for result in merged] == ["A", "AC"]


def test_merge_answer_results_rejects_unknown_override_qid():
    with pytest.raises(ValueError, match="unknown qids"):
        merge_answer_results([_result("q1", "A")], [[_result("q2", "B")]])


def test_validate_answer_results_accepts_complete_valid_submission():
    validate_answer_results(
        [_result("q1", "AC"), _result("q2", "B")],
        [_question("q1"), _question("q2", "mcq")],
        require_complete=True,
    )


@pytest.mark.parametrize("answer", ["CA", "AA", "A,C", ""])
def test_validate_answer_results_rejects_invalid_multi_answer(answer: str):
    with pytest.raises(ValueError, match="invalid answer|sorted and unique"):
        validate_answer_results([_result("q1", answer)], [_question("q1")])


def test_validate_answer_results_rejects_missing_qid_and_bad_tokens():
    with pytest.raises(ValueError, match="missing 1 qids"):
        validate_answer_results([_result("q1", "A")], [_question("q1"), _question("q2")], require_complete=True)

    with pytest.raises(ValueError, match="inconsistent token usage"):
        validate_answer_results([_result("q1", "A", total=4)], [_question("q1")])
