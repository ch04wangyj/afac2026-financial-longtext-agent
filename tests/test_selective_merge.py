from agent.evaluation.selective_merge import merge_candidate_with_baseline
from agent.schemas import AnswerResult, TokenUsage


def _result(qid: str, answer: str, tokens: int) -> AnswerResult:
    return AnswerResult(
        qid=qid,
        answer=answer,
        confidence=0.9,
        evidence=[],
        token_usage=TokenUsage(prompt_tokens=tokens),
    )


def test_selective_merge_uses_candidate_usage_but_falls_back_on_unreviewed_change():
    baseline = [_result("q1", "A", 100), _result("q2", "B", 100), _result("q3", "A", 100)]
    candidate = [_result("q1", "B", 20), _result("q2", "B", 30)]

    merged = merge_candidate_with_baseline(baseline, candidate, {})

    assert [row.answer for row in merged] == ["A", "B", "A"]
    assert [row.token_usage.total_tokens for row in merged] == [20, 30, 100]
    assert merged[0].metadata["merge_decision"] == "candidate_difference_fallback"
    assert merged[0].metadata["baseline_label"] == "baseline"


def test_selective_merge_accepts_explicit_source_review_answer():
    baseline = [_result("q1", "A", 100)]
    candidate = [_result("q1", "B", 20)]

    merged = merge_candidate_with_baseline(
        baseline,
        candidate,
        {"q1": {"answer": "BC", "reason": "direct source"}},
    )

    assert merged[0].answer == "BC"
    assert merged[0].metadata["source_review"]["reason"] == "direct source"
