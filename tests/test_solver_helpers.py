"""Solver 辅助函数的单元测试。"""

import unittest

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.solver import Solver, _limit_evidence_with_doc_coverage
from agent.runtime.logicrag_config import ThinkingProfile
from agent.schemas import Question, RetrievalResult, TokenUsage


def build_thinking_profiles() -> dict[str, ThinkingProfile]:
    return {
        "answer_single_pass": ThinkingProfile(enabled=False, max_tokens=384),
        "logicrag_planner": ThinkingProfile(enabled=True, max_tokens=1024),
        "logicrag_rank_summary": ThinkingProfile(enabled=True, max_tokens=640),
        "logicrag_final_compose": ThinkingProfile(enabled=True, max_tokens=1024),
        "option_judgement": ThinkingProfile(enabled=False, max_tokens=192),
        "multi_option_fallback": ThinkingProfile(enabled=True, max_tokens=512),
    }


def rr(chunk_id: str, doc_id: str) -> RetrievalResult:
    """构造测试用证据。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_contracts",
        score=1.0,
        source="test",
        query="",
        evidence_text="证据",
    )


class SolverHelpersTest(unittest.TestCase):
    def test_limit_evidence_keeps_doc_coverage(self):
        evidence = [rr("a1", "doc1"), rr("a2", "doc1"), rr("b1", "doc2")]
        selected = _limit_evidence_with_doc_coverage(evidence, top_k=2, doc_ids=["doc1", "doc2"])
        self.assertEqual([item.doc_id for item in selected], ["doc1", "doc2"])

    def test_multi_option_judgement_uses_low_budget_profile(self):
        question = Question(
            qid="q_multi",
            domain="insurance",
            split="a",
            question="哪些选项正确？",
            options={"A": "选项A", "B": "选项B"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        llm = FakeLLM(
            [
                '{"verdict":true,"confidence":0.9}',
                '{"verdict":false,"confidence":0.7}',
            ]
        )
        solver = Solver(FakeRetriever(), RuleEvidenceCompressor(max_chars=1000, top_k=2), llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "A")
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(all(call["max_tokens"] == 192 for call in llm.calls))
        self.assertTrue(all(call["enable_thinking"] is False for call in llm.calls))

    def test_multi_all_false_triggers_single_pass_fallback(self):
        question = Question(
            qid="q1",
            domain="insurance",
            split="a",
            question="哪些选项正确？",
            options={"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        llm = FakeLLM(
            [
                '{"verdict":false,"confidence":0.9}',
                '{"verdict":false,"confidence":0.9}',
                '{"verdict":false,"confidence":0.9}',
                '{"verdict":false,"confidence":0.9}',
                '{"answer":"C","confidence":0.8}',
            ]
        )
        solver = Solver(FakeRetriever(), RuleEvidenceCompressor(max_chars=1000, top_k=2), llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "C")
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.metadata["fallback"]["strategy"], "single_pass_after_all_false")
        self.assertEqual(result.token_usage.total_tokens, 10)
        self.assertEqual([call["max_tokens"] for call in llm.calls[:4]], [192, 192, 192, 192])
        self.assertTrue(all(call["enable_thinking"] is False for call in llm.calls[:4]))
        self.assertEqual(llm.calls[-1]["max_tokens"], 512)
        self.assertTrue(llm.calls[-1]["enable_thinking"] is True)


class FakeRetriever:
    """返回固定证据，避免测试依赖真实索引。"""

    def retrieve(self, question):
        return [rr(f"{question.qid}:chunk", "doc1")]


class FakeLLM:
    """按队列返回模型响应，用于验证 Solver 控制流。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []
        self.settings = Settings(enable_multi_option_judgement=True)

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        text = self.responses.pop(0)
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=1, completion_tokens=1))


if __name__ == "__main__":
    unittest.main()
