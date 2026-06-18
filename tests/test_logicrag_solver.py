from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.llm.qwen_client import LLMResponse
from agent.reasoning import logicrag
from agent.reasoning.solver import Solver
from agent.runtime.logicrag_config import ABoardRuntimeConfig, ConcurrencyConfig, LogicRAGRuntimeConfig, LogicRAGSection, QwenRuntimeConfig, ThinkingProfile
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



def build_runtime_config(*, option_matrix_enabled: bool = False, coverage_gate_enabled: bool = False) -> LogicRAGRuntimeConfig:
    return LogicRAGRuntimeConfig(
        qwen=QwenRuntimeConfig(),
        thinking_profiles=build_thinking_profiles(),
        logicrag=LogicRAGSection(),
        a_board=ABoardRuntimeConfig(
            option_matrix_enabled=option_matrix_enabled,
            coverage_gate_enabled=coverage_gate_enabled,
            force_doc_coverage_for_a_board=True,
            use_doc_ids_as_hint_only=False,
            financial_calculator_enabled=False,
        ),
        concurrency=ConcurrencyConfig(),
    )



class LogicRAGSolverTest(unittest.TestCase):
    def test_logicrag_agent_returns_compact_rank_memories(self):
        question = Question(
            qid="q_logicrag",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={
                "A": "不需要识别受益所有人",
                "B": "需要识别受益所有人并保存身份资料",
            },
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=True,
            logicrag_max_subproblems=4,
            logicrag_max_ranks=3,
            logicrag_rank_top_k=2,
            logicrag_memory_chars=800,
            logicrag_plan_max_tokens=256,
            logicrag_summary_max_tokens=128,
            answer_max_tokens=128,
        )
        solver = Solver(
            FakeLogicRetriever(),
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            FakeLogicLLM(settings),
        )

        result = solver.solve(question)

        self.assertEqual(result.answer, "B")
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(result.token_usage.total_tokens, 0)
        self.assertEqual(result.metadata["strategy"], "logicrag_agent")
        self.assertEqual(result.metadata["logic_plan"]["nodes"][0]["node_id"], "n1")
        self.assertEqual([item["rank"] for item in result.metadata["rank_memories"]], [0, 1])
        self.assertEqual(len(result.evidence), 2)
        self.assertIn("受益所有人", result.metadata["rank_memories"][0]["summary"])
        self.assertEqual(
            set(result.metadata["rank_memories"][0].keys()),
            {"rank", "summary", "evidence_doc_ids"},
        )

    def test_logicrag_agent_uses_rank_memory_in_later_queries(self):
        question = Question(
            qid="q_logicrag",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={
                "A": "不需要识别受益所有人",
                "B": "需要识别受益所有人并保存身份资料",
            },
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=True,
            logicrag_max_subproblems=4,
            logicrag_max_ranks=3,
            logicrag_rank_top_k=2,
            logicrag_memory_chars=800,
            logicrag_plan_max_tokens=256,
            logicrag_summary_max_tokens=128,
            answer_max_tokens=128,
        )
        retriever = FakeLogicRetriever()
        solver = Solver(
            retriever,
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            FakeLogicLLM(settings),
        )

        solver.solve(question)

        rank1_queries = [query for source, query in retriever.index.calls if source == "logicrag_agent_rank_1"]
        self.assertTrue(rank1_queries)
        self.assertEqual(len(rank1_queries), len(set(rank1_queries)))
        self.assertEqual(len(rank1_queries), 3)
        self.assertEqual(sum("上游记忆锚点" in query for query in rank1_queries), 1)

    def test_logicrag_agent_respects_enable_flag(self):
        question = Question(
            qid="q_logicrag",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=False,
            answer_max_tokens=128,
        )
        retriever = FakeLogicRetriever()
        solver = Solver(
            retriever,
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            FakeStandardLLM(settings),
        )

        result = solver.solve(question)

        self.assertEqual(result.answer, "B")
        self.assertNotIn("strategy", result.metadata)
        self.assertEqual([source for source, _query in retriever.index.calls], ["test"])

    def test_build_rankwise_queries_does_not_mutate_while_iterating(self):
        question = Question(
            qid="q_logicrag",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        group = {
            "rank": 1,
            "nodes": [],
            "queries": ["Q1", "Q2"],
        }

        queries = logicrag.build_rankwise_queries_for_group(
            question,
            group,
            prior_memories=[{"rank": 0, "summary": "上游记忆锚点"}],
        )

        self.assertEqual(len(queries), 3)
        self.assertEqual(queries[:2], ["Q1", "Q2"])
        self.assertEqual(sum("上游记忆锚点" in query for query in queries), 1)

    def test_logicrag_agent_enables_thinking_for_every_qwen_call(self):
        question = Question(
            qid="q_logicrag",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={
                "A": "不需要识别受益所有人",
                "B": "需要识别受益所有人并保存身份资料",
            },
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=True,
            logicrag_max_subproblems=4,
            logicrag_max_ranks=3,
            logicrag_rank_top_k=2,
            logicrag_memory_chars=800,
            logicrag_plan_max_tokens=256,
            logicrag_summary_max_tokens=128,
            answer_max_tokens=128,
        )
        fake_llm = FakeLogicLLM(settings)
        solver = Solver(
            FakeLogicRetriever(),
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            fake_llm,
        )

        solver.solve(question)

        self.assertGreaterEqual(len(fake_llm.calls), 3)
        self.assertTrue(all(call["enable_thinking"] is True for call in fake_llm.calls))
        self.assertEqual(
            [call["max_tokens"] for call in fake_llm.calls],
            [1024, 640, 640, 1024],
        )

    def test_standard_answer_path_uses_low_budget_profile(self):
        question = Question(
            qid="q_standard",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=False,
            answer_max_tokens=128,
        )
        fake_llm = FakeStandardLLM(settings)
        solver = Solver(
            FakeLogicRetriever(),
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            fake_llm,
        )

        result = solver.solve(question)

        self.assertEqual(result.answer, "B")
        self.assertEqual(len(fake_llm.calls), 1)
        self.assertEqual(fake_llm.calls[0]["max_tokens"], 384)
        self.assertFalse(fake_llm.calls[0]["enable_thinking"])

    def test_solver_uses_option_matrix_when_enabled(self):
        question = Question(
            qid="q_matrix",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降", "C": "研发费用增加"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="hybrid",
            logicrag_enabled=False,
            answer_max_tokens=128,
        )
        fake_llm = FakeOptionMatrixLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config(option_matrix_enabled=True)):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "AC")
        self.assertEqual(result.metadata["strategy"], "option_matrix")
        self.assertEqual(set(result.metadata["option_verdicts"].keys()), {"A", "B", "C"})

    def test_option_matrix_metadata_records_option_candidate_counts(self):
        question = Question(
            qid="q_matrix_cov",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "现金流增长"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="hybrid", logicrag_enabled=False)
        fake_llm = FakeOptionMatrixLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config(option_matrix_enabled=True)):
            solver = Solver(OptionCoverageRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.metadata["option_coverage"]["B"]["candidate_count"], 0)
        self.assertTrue(result.metadata["option_coverage"]["B"]["missing"])


class FakeLogicRetriever:
    def __init__(self) -> None:
        self.index = FakeIndex()
        self.doc_index = None
        self.top_k_per_query = 2
        self.fused_top_k = 3
        self.strategy = "logicrag_agent"
        self.blind_top_docs = 4

    @staticmethod
    def _question_with_options(question):
        return f"{question.question} " + " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))

    def _candidate_doc_filter(self, question, restrict_to_doc_ids=True):
        return set(question.doc_ids) if restrict_to_doc_ids and question.doc_ids else None

    def retrieve(self, question, restrict_to_doc_ids=True):
        return self.index.search(question.question, top_k=self.fused_top_k, filter_doc_ids=self._candidate_doc_filter(question))


class OptionCoverageRetriever(FakeLogicRetriever):
    def __init__(self) -> None:
        self.index = OptionCoverageIndex()
        self.doc_index = None
        self.top_k_per_query = 2
        self.fused_top_k = 3
        self.strategy = "hybrid"
        self.blind_top_docs = 4


class FakeIndex:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=2, filter_doc_ids=None, source="test"):
        self.calls.append((source, query))
        base = [
            RetrievalResult(
                chunk_id="c1",
                doc_id="doc1",
                domain="regulatory",
                score=2.0,
                source=source,
                query=query,
                evidence_text="办法要求银行识别受益所有人并保存客户身份资料。",
                metadata={"page": 1, "title": "客户尽职调查办法"},
            ),
            RetrievalResult(
                chunk_id="c2",
                doc_id="doc1",
                domain="regulatory",
                score=1.5,
                source=source,
                query=query,
                evidence_text="2024年相关义务继续执行，开户时应核验并留存资料。",
                metadata={"page": 2, "title": "客户尽职调查办法"},
            ),
        ]
        return base[:top_k]


class OptionCoverageIndex:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=2, filter_doc_ids=None, source="test"):
        self.calls.append((source, query))
        if "现金流" in query:
            return []
        return [
            RetrievalResult(
                chunk_id="oa1",
                doc_id="doc1",
                domain="financial_reports",
                score=1.0,
                source=source,
                query=query,
                evidence_text="2025 年营业收入同比增长 12%，研发费用同比增加 5%。",
                metadata={"page": 3, "title": "2025年年报"},
            )
        ]


class FakeLogicLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []
        self.profiles = build_thinking_profiles()

    def chat(
        self,
        messages,
        temperature=0.0,
        max_tokens=0,
        enable_thinking=None,
        thinking_profile=None,
    ):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        if "LogicRAG规划器" in prompt:
            return LLMResponse(
                text='{"subproblems":[{"id":"n1","text":"定位受益所有人识别义务","depends_on":[]},{"id":"n2","text":"确认是否需要保存客户身份资料","depends_on":["n1"]}],"rationale":"先定位义务，再确认留存要求。"}',
                usage=TokenUsage(prompt_tokens=5, completion_tokens=4),
            )
        if "LogicRAG memory summary" in prompt:
            if "rank=0" in prompt:
                text = "受益所有人识别义务：银行必须识别受益所有人。上游记忆锚点"
            else:
                text = "资料留存义务：银行还需保存客户身份资料。"
            return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=4, completion_tokens=3))
        if "LogicRAG final compose" in prompt:
            return LLMResponse(
                text='{"answer":"B","confidence":0.86,"reason":"证据明确要求识别受益所有人并保存身份资料"}',
                usage=TokenUsage(prompt_tokens=4, completion_tokens=4),
            )
        raise AssertionError(f"unexpected prompt: {prompt}")


class FakeStandardLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []
        self.profiles = build_thinking_profiles()

    def chat(
        self,
        messages,
        temperature=0.0,
        max_tokens=0,
        enable_thinking=None,
        thinking_profile=None,
    ):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        return LLMResponse(
            text='{"answer":"B","confidence":0.72,"reason":"证据显示需要识别受益所有人并保存身份资料"}',
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
        )


class FakeOptionMatrixLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        if "2025 年收入增长" in prompt:
            text = '{"option":"A","relation":"support","confidence":0.95,"support_evidence":["[1]"],"reason":"收入同比增长"}'
        elif "2025 年收入下降" in prompt:
            text = '{"option":"B","relation":"refute","confidence":0.92,"refute_evidence":["[1]"],"reason":"收入并未下降"}'
        elif "研发费用增加" in prompt:
            text = '{"option":"C","relation":"support","confidence":0.88,"support_evidence":["[1]"],"reason":"研发费用增加"}'
        else:
            text = '{"option":"B","relation":"insufficient","confidence":0.10,"reason":"证据不足"}'
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=2, completion_tokens=2))


if __name__ == "__main__":
    unittest.main()
