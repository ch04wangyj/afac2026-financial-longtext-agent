"""Solver 辅助函数的单元测试。"""

import unittest

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.solver import Solver, _limit_evidence_with_doc_coverage
from agent.schemas import Question, RetrievalResult, TokenUsage


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


class FakeRetriever:
    """返回固定证据，避免测试依赖真实索引。"""

    def retrieve(self, question):
        return [rr(f"{question.qid}:chunk", "doc1")]


class FakeLLM:
    """按队列返回模型响应，用于验证 Solver 控制流。"""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.settings = Settings(enable_multi_option_judgement=True)

    def chat(self, *args, **kwargs):
        text = self.responses.pop(0)
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=1, completion_tokens=1))


if __name__ == "__main__":
    unittest.main()
