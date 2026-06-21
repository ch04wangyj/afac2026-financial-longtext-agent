from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.llm.qwen_client import LLMResponse
from agent.reasoning import logicrag
from agent.reasoning.answer_parser import extract_json_object
from agent.reasoning.solver import Solver
from agent.runtime.logicrag_config import ABoardRuntimeConfig, ConcurrencyConfig, LogicRAGRuntimeConfig, LogicRAGSection, QwenRuntimeConfig, ThinkingProfile
from agent.schemas import Question, RetrievalResult, TokenUsage
from agent.domain.coverage_rules import expected_evidence_facets


def build_thinking_profiles() -> dict[str, ThinkingProfile]:
    return {
        "answer_single_pass": ThinkingProfile(enabled=False, max_tokens=384),
        "logicrag_planner": ThinkingProfile(enabled=True, max_tokens=1024),
        "logicrag_rank_summary": ThinkingProfile(enabled=True, max_tokens=640),
        "logicrag_final_compose": ThinkingProfile(enabled=True, max_tokens=1024),
        "option_judgement": ThinkingProfile(enabled=False, max_tokens=192),
        "multi_option_fallback": ThinkingProfile(enabled=True, max_tokens=512),
    }



def build_runtime_config(
    *,
    option_matrix_enabled: bool = False,
    multi_logicrag_enabled: bool = True,
    multi_logicrag_retry_enabled: bool = True,
    coverage_gate_enabled: bool = False,
    claim_centric_multi_enabled: bool = False,
    claim_centric_mcq_enabled: bool = False,
    max_claim_refinement_rounds: int = 1,
    max_claim_query_bundles: int = 5,
    claim_final_compose_enabled: bool = False,
    claim_verdict_max_evidence_items: int = 6,
    claim_retry_verdict_max_evidence_items: int = 8,
) -> LogicRAGRuntimeConfig:
    return LogicRAGRuntimeConfig(
        qwen=QwenRuntimeConfig(),
        thinking_profiles=build_thinking_profiles(),
        logicrag=LogicRAGSection(),
        a_board=ABoardRuntimeConfig(
            option_matrix_enabled=option_matrix_enabled,
            multi_logicrag_enabled=multi_logicrag_enabled,
            multi_logicrag_retry_enabled=multi_logicrag_retry_enabled,
            coverage_gate_enabled=coverage_gate_enabled,
            force_doc_coverage_for_a_board=True,
            use_doc_ids_as_hint_only=False,
            financial_calculator_enabled=False,
            claim_centric_multi_enabled=claim_centric_multi_enabled,
            claim_centric_mcq_enabled=claim_centric_mcq_enabled,
            max_claim_refinement_rounds=max_claim_refinement_rounds,
            max_claim_query_bundles=max_claim_query_bundles,
            claim_final_compose_enabled=claim_final_compose_enabled,
            claim_verdict_max_evidence_items=claim_verdict_max_evidence_items,
            claim_retry_verdict_max_evidence_items=claim_retry_verdict_max_evidence_items,
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
        self.assertEqual(len(rank1_queries), 2)
        self.assertGreaterEqual(sum("必须识别受益所有人" in query or "受益所有人识别义务" in query for query in rank1_queries), 1)

    def test_logicrag_final_compose_uses_last_rank_evidence_instead_of_accumulated_raw_context(self):
        question = Question(
            qid="q_logicrag_last_rank",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
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
            ProgressiveEvidenceRetriever(),
            RuleEvidenceCompressor(max_chars=1200, top_k=3),
            fake_llm,
        )

        result = solver.solve(question)

        final_prompt = fake_llm.prompts[-1]
        self.assertEqual(result.answer, "B")
        self.assertIn("第二层证据：银行还需保存客户身份资料。", final_prompt)
        self.assertNotIn("第一层证据：银行必须识别受益所有人。", final_prompt)
        self.assertTrue(any("第一层证据：银行必须识别受益所有人。" in item.evidence_text for item in result.evidence))

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

        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[0], "Q1")
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

    def test_solver_uses_multi_logicrag_route_for_multi_questions(self):
        question = Question(
            qid="q_multi_logicrag",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降", "C": "研发费用增加"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=True,
            answer_max_tokens=128,
        )
        fake_llm = FakeMultiLogicLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config()):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "AC")
        self.assertEqual(result.metadata["strategy"], "multi_logicrag")
        self.assertEqual(set(result.metadata["option_runs"].keys()), {"A", "B", "C"})

    def test_multi_logicrag_expands_every_uncertain_option(self):
        question = Question(
            qid="q_multi_retry",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(
            retrieval_strategy="logicrag_agent",
            logicrag_enabled=True,
            answer_max_tokens=128,
        )
        retriever = RetryAwareRetriever()
        fake_llm = FakeRetryMultiLogicLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config()):
            solver = Solver(retriever, RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "AB")
        self.assertFalse(result.metadata["option_runs"]["A"]["retried"])
        self.assertTrue(result.metadata["option_runs"]["B"]["retried"])
        self.assertTrue(any(source == "multi_logicrag_retry_B" for source, _query in retriever.index.calls))
        self.assertIn("sufficiency", result.metadata["option_runs"]["B"])
        self.assertIn("comparison_incomplete", result.metadata["option_runs"]["B"]["sufficiency"])
        self.assertTrue(
            any("2025" in query and "收入下降" in query for query in result.metadata["option_runs"]["B"]["retry_queries"])
        )
        self.assertIn("refinement_triggered", result.metadata["option_runs"]["B"])

    def test_solver_uses_claim_centric_multi_route_when_enabled(self):
        question = Question(
            qid="q_claim_multi",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降", "C": "研发费用增加"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="logicrag_agent", logicrag_enabled=True, answer_max_tokens=128)
        fake_llm = FakeMultiLogicLLM(settings)
        with patch(
            "agent.reasoning.solver.load_logicrag_runtime_config",
            return_value=build_runtime_config(claim_centric_multi_enabled=True),
        ):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "AC")
        self.assertEqual(result.metadata["strategy"], "claim_centric_multi")
        self.assertEqual(set(result.metadata["claim_runs"].keys()), {"A", "B", "C"})
        self.assertEqual(result.metadata["answer_assembly_policy"], "set_verified_exact_match")
        self.assertIn("token_budget_policy", result.metadata)
        self.assertFalse(result.metadata["token_budget_policy"]["claim_final_compose_enabled"])

    def test_solver_uses_claim_centric_mcq_route_when_enabled(self):
        question = Question(
            qid="q_claim_mcq",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="logicrag_agent", logicrag_enabled=True, answer_max_tokens=128)
        fake_llm = FakeClaimMcqLLM(settings)
        with patch(
            "agent.reasoning.solver.load_logicrag_runtime_config",
            return_value=build_runtime_config(claim_centric_mcq_enabled=True),
        ):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.answer, "B")
        self.assertEqual(result.metadata["strategy"], "claim_centric_mcq")
        self.assertEqual(result.metadata["answer_assembly_policy"], "set_verified_exact_match")
        self.assertEqual(set(result.metadata["claim_runs"].keys()), {"A", "B"})

    def test_claim_centric_route_records_required_claim_run_metadata(self):
        question = Question(
            qid="q_claim_meta",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="logicrag_agent", logicrag_enabled=True, answer_max_tokens=128)
        fake_llm = FakeRetryMultiLogicLLM(settings)
        with patch(
            "agent.reasoning.solver.load_logicrag_runtime_config",
            return_value=build_runtime_config(claim_centric_multi_enabled=True),
        ):
            solver = Solver(RetryAwareRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        required = {
            "claim_id",
            "option_key",
            "claim_type",
            "query_bundles",
            "evidence_doc_ids",
            "evidence_chunk_ids",
            "sufficiency",
            "retried",
            "refinement",
            "relation",
            "confidence",
        }
        self.assertTrue(required.issubset(set(result.metadata["claim_runs"]["A"].keys())))

    def test_claim_centric_route_records_token_budget_policy(self):
        question = Question(
            qid="q_claim_budget",
            domain="financial_reports",
            split="A",
            question="根据财报披露，以下哪些说法正确？",
            options={"A": "2025 年收入增长", "B": "2025 年收入下降"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="logicrag_agent", logicrag_enabled=True, answer_max_tokens=128)
        fake_llm = FakeRetryMultiLogicLLM(settings)
        with patch(
            "agent.reasoning.solver.load_logicrag_runtime_config",
            return_value=build_runtime_config(claim_centric_multi_enabled=True),
        ):
            solver = Solver(RetryAwareRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertIs(result.metadata["token_budget_policy"]["claim_final_compose_enabled"], False)
        self.assertLessEqual(result.metadata["token_budget_policy"]["max_claim_refinement_rounds"], 1)

    def test_logicrag_agent_records_rank_sufficiency_report(self):
        question = Question(
            qid="q_logicrag_sufficiency",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
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
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config()):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertIn("rank_runs", result.metadata)
        self.assertIn("sufficiency", result.metadata["rank_runs"][0])

    def test_financial_calculator_enabled_records_metric_extraction_metadata(self):
        question = Question(
            qid="fin_calc_meta",
            domain="financial_reports",
            split="A",
            question="根据财报披露，下列关于现金流与收入比例的说法哪些正确？",
            options={"A": "比亚迪2024年经营活动现金流净额低于营业收入的一半"},
            answer_format="multi",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="hybrid", logicrag_enabled=False, enable_multi_option_judgement=False)
        fake_llm = FakeFinancialExtractionLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config()):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)
        solver.runtime = LogicRAGRuntimeConfig(
            qwen=solver.runtime.qwen,
            thinking_profiles=solver.runtime.thinking_profiles,
            logicrag=solver.runtime.logicrag,
            a_board=ABoardRuntimeConfig(
                option_matrix_enabled=False,
                coverage_gate_enabled=False,
                force_doc_coverage_for_a_board=True,
                use_doc_ids_as_hint_only=False,
                financial_calculator_enabled=True,
            ),
            concurrency=solver.runtime.concurrency,
            source_path=solver.runtime.source_path,
        )

        result = solver.solve(question)

        self.assertIn("financial_metric_extraction", result.metadata)
        parsed = result.metadata["financial_metric_extraction"]
        self.assertEqual(parsed["metric_values"][0]["metric"], "营业收入")

    def test_solver_metadata_includes_domain_coverage_facets_when_gate_enabled(self):
        question = Question(
            qid="facet_meta",
            domain="financial_reports",
            split="A",
            question="比较比亚迪2024年和美的2025年营业收入与现金流。",
            options={"A": "收入增长"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )
        settings = Settings(retrieval_strategy="hybrid", logicrag_enabled=False, enable_multi_option_judgement=False)
        fake_llm = FakeStandardLLM(settings)
        with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=build_runtime_config(coverage_gate_enabled=True)):
            solver = Solver(FakeLogicRetriever(), RuleEvidenceCompressor(max_chars=1200, top_k=3), fake_llm)

        result = solver.solve(question)

        self.assertEqual(result.metadata["domain_coverage_facets"], expected_evidence_facets("financial_reports", question.question))


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


class RetryAwareRetriever(FakeLogicRetriever):
    def __init__(self) -> None:
        self.index = RetryAwareIndex()
        self.doc_index = None
        self.top_k_per_query = 2
        self.fused_top_k = 3
        self.strategy = "logicrag_agent"
        self.blind_top_docs = 4


class ProgressiveEvidenceRetriever(FakeLogicRetriever):
    def __init__(self) -> None:
        self.index = ProgressiveEvidenceIndex()
        self.doc_index = None
        self.top_k_per_query = 2
        self.fused_top_k = 3
        self.strategy = "logicrag_agent"
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


class RetryAwareIndex:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=2, filter_doc_ids=None, source="test"):
        self.calls.append((source, query))
        if source == "multi_logicrag_retry_B":
            return [
                RetrievalResult(
                    chunk_id="rb1",
                    doc_id="doc1",
                    domain="financial_reports",
                    score=1.2,
                    source=source,
                    query=query,
                    evidence_text="2025 年收入并未下降，而是同比增长 12%。",
                    metadata={"page": 4, "title": "2025年年报"},
                )
            ]
        return [
            RetrievalResult(
                chunk_id=f"seed:{abs(hash((source, query))) % 10000}",
                doc_id="doc1",
                domain="financial_reports",
                score=1.0,
                source=source,
                query=query,
                evidence_text="2025 年营业收入同比增长 12%，研发费用同比增加 5%。",
                metadata={"page": 3, "title": "2025年年报"},
            )
        ]


class ProgressiveEvidenceIndex:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=2, filter_doc_ids=None, source="test"):
        self.calls.append((source, query))
        if source == "logicrag_agent_rank_1":
            return [
                RetrievalResult(
                    chunk_id="p2",
                    doc_id="doc1",
                    domain="regulatory",
                    score=1.8,
                    source=source,
                    query=query,
                    evidence_text="第二层证据：银行还需保存客户身份资料。",
                    metadata={"page": 2, "title": "客户尽职调查办法"},
                )
            ]
        return [
            RetrievalResult(
                chunk_id="p1",
                doc_id="doc1",
                domain="regulatory",
                score=2.0,
                source=source,
                query=query,
                evidence_text="第一层证据：银行必须识别受益所有人。",
                metadata={"page": 1, "title": "客户尽职调查办法"},
            )
        ]


class FakeLogicLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []
        self.prompts = []
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
        self.prompts.append(prompt)
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


class FakeMultiLogicLLM:
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
            raise AssertionError(f"unexpected prompt: {prompt}")
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=2, completion_tokens=2))


class FakeClaimMcqLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        if "不需要识别受益所有人" in prompt:
            text = '{"option":"A","relation":"refute","confidence":0.90,"refute_evidence":["[1]"],"reason":"法规要求识别受益所有人"}'
        elif "需要识别受益所有人并保存身份资料" in prompt:
            text = '{"option":"B","relation":"support","confidence":0.96,"support_evidence":["[1]","[2]"],"reason":"法规明确要求识别并留存资料"}'
        else:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=2, completion_tokens=2))


class FakeRetryMultiLogicLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []
        self.option_counts = {"A": 0, "B": 0}

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        if "2025 年收入增长" in prompt:
            self.option_counts["A"] += 1
            text = '{"option":"A","relation":"support","confidence":0.95,"support_evidence":["[1]"],"reason":"收入同比增长"}'
        elif "2025 年收入下降" in prompt:
            self.option_counts["B"] += 1
            if self.option_counts["B"] == 1:
                text = '{"option":"B","relation":"insufficient","confidence":0.35,"support_evidence":[],"refute_evidence":[],"reason":"初次证据不足"}'
            else:
                text = '{"option":"B","relation":"support","confidence":0.85,"support_evidence":["[2]"],"refute_evidence":[],"reason":"扩检后确认正确"}'
        else:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=2, completion_tokens=2))


class FakeFinancialExtractionLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls = []

    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        if "比亚迪2024年" in prompt:
            text = '{"metric_values":[{"entity":"比亚迪","year":"2024","metric":"营业收入","value":"777,102,000","unit":"千元","evidence_id":"[1]"}],"missing_metrics":[]}'
        else:
            text = '{"metric_values":[{"entity":"美的集团","year":"2025","metric":"营业收入","value":"456,451,731","unit":"千元","evidence_id":"[1]"}],"missing_metrics":[]}'
        return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=2, completion_tokens=2))



class AdaptiveLogicRetriever(FakeLogicRetriever):
    def __init__(self) -> None:
        self.index = AdaptiveLogicIndex()
        self.doc_index = None
        self.top_k_per_query = 2
        self.fused_top_k = 3
        self.strategy = "logicrag_agent"
        self.blind_top_docs = 4


class AdaptiveLogicIndex:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, top_k=2, filter_doc_ids=None, source="test"):
        self.calls.append((source, query, set(filter_doc_ids) if filter_doc_ids else None))
        if "具体后果条款" in query:
            return [
                RetrievalResult(
                    chunk_id="adaptive_specific",
                    doc_id="doc1",
                    domain="regulatory",
                    score=2.0,
                    source=source,
                    query=query,
                    evidence_text="具体后果条款：银行必须识别受益所有人并保存客户身份资料。",
                    metadata={"page": 5, "title": "客户尽职调查办法"},
                )
            ]
        return [
            RetrievalResult(
                chunk_id="adaptive_generic",
                doc_id="doc1",
                domain="regulatory",
                score=1.0,
                source=source,
                query=query,
                evidence_text="泛页：本办法说明客户尽职调查的一般原则。",
                metadata={"page": 1, "title": "客户尽职调查办法"},
            )
        ]


class FakeAdaptiveLogicLLM(FakeLogicLLM):
    def chat(self, messages, temperature=0.0, max_tokens=0, enable_thinking=None, thinking_profile=None):
        if thinking_profile is not None:
            max_tokens = thinking_profile.max_tokens
            enable_thinking = thinking_profile.enabled
        self.calls.append({"max_tokens": max_tokens, "enable_thinking": enable_thinking})
        prompt = "\n".join(message.get("content", "") for message in messages)
        self.prompts.append(prompt)
        if "LogicRAG规划器" in prompt:
            return LLMResponse(
                text='{"subproblems":[{"id":"n1","text":"定位受益所有人识别义务","depends_on":[]}],"rationale":"定位义务。"}',
                usage=TokenUsage(prompt_tokens=5, completion_tokens=4),
            )
        if "直接提出检索组合" in prompt:
            return LLMResponse(
                text='{"query_bundles":[{"query":"受益所有人 一般原则","intent":"find_rule_context","evidence_type":"definition"}]}',
                usage=TokenUsage(prompt_tokens=4, completion_tokens=3),
            )
        if "证据是否足够" in prompt or "证据充分性判断" in prompt or "是否足够回答" in prompt:
            if "具体后果条款" in prompt:
                text = '{"sufficient":1,"failure_tags":[],"reason":"已找到具体义务和留存要求","missing_evidence":"","next_search_goal":""}'
            else:
                text = '{"sufficient":0,"failure_tags":["generic_context_only"],"reason":"只有泛页","missing_evidence":"缺少具体后果条款","next_search_goal":"查找具体后果条款"}'
            return LLMResponse(text=text, usage=TokenUsage(prompt_tokens=3, completion_tokens=2))
        if "重新思考下一轮检索方向" in prompt:
            return LLMResponse(
                text='{"query_bundles":[{"query":"受益所有人 具体后果条款 保存客户身份资料","intent":"find_clause_consequence","evidence_type":"clause_consequence"}]}',
                usage=TokenUsage(prompt_tokens=4, completion_tokens=3),
            )
        if "LogicRAG memory summary" in prompt:
            return LLMResponse(text="已找到具体后果条款和资料保存要求。", usage=TokenUsage(prompt_tokens=4, completion_tokens=3))
        if "LogicRAG final compose" in prompt:
            return LLMResponse(
                text='{"answer":"B","confidence":0.86,"reason":"证据明确要求识别受益所有人并保存身份资料"}',
                usage=TokenUsage(prompt_tokens=4, completion_tokens=4),
            )
        raise AssertionError(f"unexpected prompt: {prompt}")


def test_logicrag_adaptive_retrieval_refines_when_sufficiency_fails():
    question = Question(
        qid="q_logicrag_adaptive",
        domain="regulatory",
        split="A",
        question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
        options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
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
    runtime = LogicRAGRuntimeConfig(
        qwen=QwenRuntimeConfig(),
        thinking_profiles=build_thinking_profiles(),
        logicrag=LogicRAGSection(adaptive_retrieval_enabled=True, max_refinement_rounds_per_rank=1),
        a_board=ABoardRuntimeConfig(use_doc_ids_as_hint_only=False),
        concurrency=ConcurrencyConfig(),
    )
    retriever = AdaptiveLogicRetriever()
    with patch("agent.reasoning.solver.load_logicrag_runtime_config", return_value=runtime):
        solver = Solver(retriever, RuleEvidenceCompressor(max_chars=1200, top_k=3), FakeAdaptiveLogicLLM(settings))

    result = solver.solve(question)

    adaptive = result.metadata["rank_runs"][0]["adaptive_retrieval"]
    assert result.answer == "B"
    assert adaptive["rounds"][0]["sufficiency"]["sufficient"] is False
    assert adaptive["rounds"][1]["queries"] == ["受益所有人 具体后果条款 保存客户身份资料"]
    assert adaptive["exhausted"] is False
    assert any(source == "logicrag_agent_rank_0_round_1" for source, _query, _scope in retriever.index.calls)
    assert all(scope == {"doc1"} for _source, _query, scope in retriever.index.calls)


if __name__ == "__main__":
    unittest.main()
