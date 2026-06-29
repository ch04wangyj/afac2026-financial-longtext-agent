import json
from pathlib import Path

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


def test_v5_merge_keeps_candidate_token_but_requires_review_for_answer_change():
    baseline = [_result("q1", "A", 100), _result("q2", "B", 100)]
    candidate = [_result("q1", "B", 20), _result("q2", "A", 30)]

    merged = merge_candidate_with_baseline(
        baseline,
        candidate,
        {"q2": {"answer": "A", "reason": "原文直接证据"}},
        baseline_label="v4_layout",
        candidate_label="v5_structure",
    )

    assert [row.answer for row in merged] == ["A", "A"]
    assert [row.token_usage.total_tokens for row in merged] == [20, 30]
    assert merged[0].metadata["merge_decision"] == "candidate_difference_fallback"
    assert merged[1].metadata["source_review"]["reason"] == "原文直接证据"


def test_v5_review_config_is_valid_json():
    path = Path(__file__).resolve().parents[1] / "configs" / "v5_structure_reviews.json"

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload) == 14
    assert all(row.get("answer") and row.get("reason") for row in payload.values())
