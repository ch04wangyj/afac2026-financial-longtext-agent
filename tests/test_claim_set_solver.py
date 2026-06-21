"""claim-centric 集合级复核 Solver 集成测试。"""

from unittest.mock import patch

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.llm.qwen_client import LLMResponse
from agent.reasoning.solver import Solver
from agent.runtime.logicrag_config import (
    ABoardRuntimeConfig,
    LogicRAGRuntimeConfig,
    ThinkingProfile,
)
from agent.schemas import Question, RetrievalResult, TokenUsage


def _result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id="doc1",
        domain="financial_reports",
        score=score,
        source="test",
        query="营业收入",
        evidence_text=text,
        metadata={"chunk_type": "table", "page": 1},
    )


class FakeIndex:
    def __init__(self) -> None:
        self.items = [
            _result("c1", "单位：亿元 2025年营业收入为120，2024年营业收入为100。", 10.0),
            _result("c2", "2025年研发费用为15亿元，较2024年的12亿元增加。", 8.0),
        ]

    def search(self, query, top_k, filter_doc_ids=None, source="test"):
        return self.items[:top_k]


class FakeRetriever:
    def __init__(self) -> None:
        self.index = FakeIndex()
        self.top_k_per_query = 5

    def _candidate_doc_filter(self, question, strict=True):
        return set(question.doc_ids)

    def _question_with_options(self, question):
        return f"{question.question} {' '.join(question.options.values())}"


class FakeLLM:
    def __init__(self) -> None:
        self.settings = Settings(retrieval_strategy="logicrag_agent", logicrag_enabled=True)
        self.prompts: list[str] = []

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        prompt = "\n".join(message.get("content", "") for message in messages)
        self.prompts.append(prompt)
        if "集合级复核" in prompt:
            text = '{"answer":"AC","confidence":0.92,"option_relations":{"A":"support","B":"refute","C":"support"},"reason":"数值账本与原文一致"}'
        elif "待判断选项: A." in prompt:
            text = '{"option":"A","relation":"support","confidence":0.9,"support_evidence":["[1]"],"refute_evidence":[],"reason":"收入增长"}'
        elif "待判断选项: B." in prompt:
            text = '{"option":"B","relation":"refute","confidence":0.9,"support_evidence":[],"refute_evidence":["[1]"],"reason":"收入未下降"}'
        elif "待判断选项: C." in prompt:
            text = '{"option":"C","relation":"support","confidence":0.8,"support_evidence":["[2]"],"refute_evidence":[],"reason":"研发费用增加"}'
        else:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=1, completion_tokens=1))


def _runtime() -> LogicRAGRuntimeConfig:
    return LogicRAGRuntimeConfig(
        thinking_profiles={
            "multi_logicrag_option_verdict": ThinkingProfile(enabled=False, max_tokens=192),
            "multi_logicrag_option_retry": ThinkingProfile(enabled=False, max_tokens=192),
            "claim_set_verification": ThinkingProfile(enabled=True, max_tokens=512),
        },
        a_board=ABoardRuntimeConfig(
            multi_logicrag_enabled=False,
            multi_logicrag_retry_enabled=False,
            claim_centric_multi_enabled=True,
            claim_centric_mcq_enabled=True,
            evidence_set_selection_enabled=True,
            claim_set_verification_enabled=True,
            numeric_fact_ledger_enabled=True,
            claim_require_valid_citations=True,
        ),
    )


def test_claim_set_solver_runs_exact_match_verification_with_fact_ledger():
    question = Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="根据财报，以下哪些说法正确？",
        options={
            "A": "2025年营业收入高于2024年",
            "B": "2025年营业收入低于2024年",
            "C": "2025年研发费用较2024年增加",
        },
        answer_format="multi",
        doc_ids=["doc1"],
    )
    llm = FakeLLM()
    with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=_runtime()):
        solver = Solver(FakeRetriever(), RuleEvidenceCompressor(max_chars=3000, top_k=4), llm)

    result = solver.solve(question)

    assert result.answer == "AC"
    assert result.confidence == 0.92
    assert result.metadata["claim_set_verification"]["triggered"] is True
    assert result.metadata["numeric_fact_ledger"]["fact_count"] >= 4
    assert result.metadata["final_evidence_selection"]["coverage_ratio"] > 0
    assert result.token_usage.total_tokens == 8
    assert any("数值事实账本" in prompt for prompt in llm.prompts)
