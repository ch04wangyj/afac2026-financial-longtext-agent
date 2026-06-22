from types import SimpleNamespace

from agent.llm.qwen_client import LLMResponse
from agent.reasoning.result_reconciler import ResultReconciler, is_weak_result
from agent.schemas import AnswerResult, Question, TokenUsage


class _FakeLLM:
    def __init__(self, answer="AC", confidence=0.9) -> None:
        self.answer = answer
        self.confidence = confidence
        self.settings = SimpleNamespace(qwen_model="fake")

    def chat(self, messages, **kwargs):
        return LLMResponse(
            text=f'{{"answer":"{self.answer}","confidence":{self.confidence}}}',
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
        )


def _question() -> Question:
    return Question(
        qid="q1",
        domain="regulatory",
        split="a",
        question="哪些说法正确？",
        options={"A": "甲", "B": "乙", "C": "丙"},
        answer_format="multi",
        doc_ids=["d1"],
    )


def _result(answer: str, raw: str) -> AnswerResult:
    return AnswerResult(
        qid="q1",
        answer=answer,
        confidence=0.9,
        evidence=[],
        token_usage=TokenUsage(prompt_tokens=20, completion_tokens=3),
        raw_response=raw,
        metadata={},
    )


def test_weak_uncertain_result_falls_back_without_extra_usage():
    current = _result("A", '{"checks":{"A":{"truth":"uncertain"},"B":{"truth":"false"},"C":{"truth":"false"}},"answer":"A"}')
    baseline = _result("AC", "{}")

    result = ResultReconciler(_FakeLLM()).reconcile(_question(), current, baseline)

    assert is_weak_result(current, _question()) is True
    assert result.answer == "AC"
    assert result.token_usage.total_tokens == 23
    assert result.metadata["reconcile_decision"] == "weak_evidence_fallback"


def test_strong_change_requires_high_confidence_confirmation():
    raw = '{"checks":{"A":{"truth":"true"},"B":{"truth":"false"},"C":{"truth":"true"}},"answer":"AC"}'
    current = _result("AC", raw)
    baseline = _result("A", "{}")

    confirmed = ResultReconciler(_FakeLLM("AC", 0.9)).reconcile(_question(), current, baseline)
    rejected = ResultReconciler(_FakeLLM("AC", 0.5)).reconcile(_question(), current, baseline)

    assert confirmed.answer == "AC"
    assert confirmed.metadata["reconcile_decision"] == "audit_confirmed_v13"
    assert rejected.answer == "A"
    assert confirmed.token_usage.total_tokens == 35
