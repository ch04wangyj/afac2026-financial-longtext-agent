"""按官网公式计算分数，并从得分反推最可能的正确题数。"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TOKEN_BUDGET = 5_000_000


@dataclass(frozen=True)
class ScoreEstimate:
    """一个正确题数假设对应的理论综合分与观测误差。"""

    correct: int
    total: int
    total_tokens: int
    expected_score: float
    observed_score: float
    absolute_error: float

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "total": self.total,
            "total_tokens": self.total_tokens,
            "expected_score": round(self.expected_score, 6),
            "observed_score": self.observed_score,
            "absolute_error": round(self.absolute_error, 6),
        }


def token_score(total_tokens: int, token_budget: int = DEFAULT_TOKEN_BUDGET) -> float:
    """实现官网 TokenScore；非正 Token 按规则计为 0。"""
    if total_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, (token_budget - total_tokens) / token_budget))


def final_score(
    correct: int,
    total: int,
    total_tokens: int,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> float:
    """计算 FinalScore = 100 * Accuracy * (0.7 + 0.3 * TokenScore)。"""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("correct/total must describe a valid evaluation set")
    efficiency = 0.7 + 0.3 * token_score(total_tokens, token_budget)
    return 100.0 * (correct / total) * efficiency


def infer_correct_count(
    observed_score: float,
    total: int,
    total_tokens: int,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> ScoreEstimate:
    """枚举离散正确题数，返回与官网显示分数最接近的一项。"""
    candidates = [
        ScoreEstimate(
            correct=correct,
            total=total,
            total_tokens=total_tokens,
            expected_score=final_score(correct, total, total_tokens, token_budget),
            observed_score=observed_score,
            absolute_error=abs(
                final_score(correct, total, total_tokens, token_budget) - observed_score
            ),
        )
        for correct in range(total + 1)
    ]
    return min(candidates, key=lambda item: (item.absolute_error, -item.correct))
