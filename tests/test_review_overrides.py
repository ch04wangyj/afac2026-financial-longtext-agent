from agent.evaluation.review_overrides import apply_review_overrides
from agent.schemas import AnswerResult, TokenUsage


def test_review_override_changes_answer_without_changing_usage():
    row = AnswerResult(
        qid="q1",
        answer="A",
        confidence=0.9,
        evidence=[],
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
    )

    result = apply_review_overrides(
        [row],
        {"q1": {"answer": "BC", "decision": "manual", "reason": "direct evidence"}},
    )[0]

    assert result.answer == "BC"
    assert result.token_usage.total_tokens == 12
    assert result.metadata["pre_review_answer"] == "A"
    assert result.metadata["final_review"]["reason"] == "direct evidence"
