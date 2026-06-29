import pytest

from agent.evaluation.score_diagnostics import (
    final_score,
    infer_correct_count,
    token_score,
)


def test_score_formula_matches_official_v5_result():
    assert final_score(82, 100, 315_727) == pytest.approx(80.446623, abs=1e-6)


def test_infer_v6_correct_count_from_displayed_score():
    estimate = infer_correct_count(83.33, 100, 326_076)

    assert estimate.correct == 85
    assert estimate.expected_score == pytest.approx(83.3370124, abs=1e-6)
    assert estimate.absolute_error < 0.01


def test_invalid_or_over_budget_tokens_follow_official_clamp():
    assert token_score(0) == 0.0
    assert token_score(6_000_000) == 0.0
