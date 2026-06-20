"""RAG 检索变体的单元测试。"""

import unittest

from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.retriever import Retriever
from agent.retrieve.variants import retrieve_with_variant
from agent.schemas import Chunk, Question


class RetrievalVariantsTest(unittest.TestCase):
    def test_oracle_restricts_to_gold_docs(self):
        chunks = [
            Chunk("c1", "gold", "insurance", 1, "", "", "身故保险金 现金价值", [], [], []),
            Chunk("c2", "other", "insurance", 1, "", "", "身故保险金 其他产品", [], [], []),
        ]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")
        question = Question(
            qid="q1",
            domain="insurance",
            split="A",
            question="身故保险金如何计算",
            options={"A": "现金价值"},
            answer_format="mcq",
            doc_ids=["gold"],
        )
        results = retrieve_with_variant(index, question, "oracle_doc_restricted", top_k=10)
        self.assertTrue(results)
        self.assertEqual({result.doc_id for result in results}, {"gold"})

    def test_bm25f_lite_rrf_prefers_number_and_clause_match(self):
        chunks = [
            Chunk(
                "c1",
                "doc-main",
                "insurance",
                1,
                "保险责任",
                "第8条",
                "本条说明一般保险责任。",
                [],
                ["20%"],
                [],
                {"title": "一般条款"},
            ),
            Chunk(
                "c2",
                "doc-main",
                "insurance",
                2,
                "保险责任",
                "第8条",
                "第8条明确现金价值按20%比例计算。",
                [],
                ["20%"],
                [],
                {"title": "现金价值计算"},
            ),
            Chunk(
                "c3",
                "doc-other",
                "insurance",
                1,
                "保险责任",
                "第3条",
                "现金价值按照10%比例计算。",
                [],
                ["10%"],
                [],
                {"title": "现金价值计算"},
            ),
        ]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")
        question = Question(
            qid="q2",
            domain="insurance",
            split="A",
            question="根据第8条，现金价值按20%如何计算？",
            options={"A": "按20%比例计算"},
            answer_format="mcq",
            doc_ids=["doc-main"],
        )

        results = retrieve_with_variant(index, question, "bm25f_lite_rrf", top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0].chunk_id, "c2")
        self.assertEqual(results[0].source, "rrf:bm25f_lite_rrf")
        self.assertIn("score_breakdown", results[0].metadata)
        self.assertEqual(results[0].metadata["score_breakdown"]["mode"], "bm25f_lite")

    def test_bm25f_lite_filter_doc_ids_isolates_normalization(self):
        allowed_chunks = [
            Chunk(
                "a1",
                "doc-allowed",
                "insurance",
                1,
                "保险责任",
                "第3条",
                "第8条 一般 说明 一般",
                [],
                ["20%"],
                [],
                {"title": "第8条说明"},
            ),
            Chunk(
                "a2",
                "doc-allowed",
                "insurance",
                2,
                "保险责任",
                "第3条",
                "计算 20% 第8条 计算 特别",
                [],
                [],
                [],
                {"title": "第8条说明"},
            ),
        ]
        query = "根据第8条，现金价值按20%如何计算？"

        without_excluded = BM25SearchIndex.build(allowed_chunks, tokenizer_mode="mixed").search(
            query,
            top_k=2,
            filter_doc_ids={"doc-allowed"},
            source="probe",
            scoring_mode="bm25f_lite",
        )

        with_excluded = BM25SearchIndex.build(
            allowed_chunks
            + [
                Chunk(
                    "b1",
                    "doc-excluded",
                    "insurance",
                    3,
                    "保险责任",
                    "第8条",
                    "20% 第8条 特别 规则 现金价值 计算",
                    [],
                    ["20%"],
                    [],
                    {"title": "第8条说明"},
                )
            ],
            tokenizer_mode="mixed",
        ).search(
            query,
            top_k=2,
            filter_doc_ids={"doc-allowed"},
            source="probe",
            scoring_mode="bm25f_lite",
        )

        self.assertEqual([result.chunk_id for result in without_excluded], ["a2", "a1"])
        self.assertEqual(
            [result.chunk_id for result in with_excluded],
            ["a2", "a1"],
            "Excluded docs must not influence normalization or ranking inside filter_doc_ids.",
        )

    def test_bm25f_lite_uses_extra_index_fields_for_company_and_metric_match(self):
        chunks = [
            Chunk(
                "c1",
                "doc-byd",
                "financial_reports",
                1,
                "管理层讨论与分析",
                "",
                "报告期内公司研发投入持续增长。",
                [],
                ["2025 年"],
                [],
                {"title": "年度报告", "extra_index_fields": ["比亚迪", "研发投入", "营业收入"]},
            ),
            Chunk(
                "c2",
                "doc-other",
                "financial_reports",
                1,
                "管理层讨论与分析",
                "",
                "报告期内公司营业收入持续增长。",
                [],
                ["2025 年"],
                [],
                {"title": "年度报告", "extra_index_fields": ["中国移动", "营业收入"]},
            ),
        ]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")

        results = index.search(
            "比亚迪 2025 年 研发投入",
            top_k=2,
            source="probe",
            scoring_mode="bm25f_lite",
        )

        self.assertTrue(results)
        self.assertEqual(results[0].doc_id, "doc-byd")
        self.assertIn("score_breakdown", results[0].metadata)
        self.assertIn("structured", results[0].metadata["score_breakdown"]["weights"])

    def test_logicrag_agent_retriever_keeps_broad_recall_default_search(self):
        chunks = [Chunk("c1", "doc1", "regulation", 1, "", "", "受益所有人 身份资料", [], [], [], {})]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")

        Retriever(index, strategy="logicrag_agent")

        self.assertEqual(index.default_search_mode, "bm25")

    def test_broad_sparse_structured_rerank_preserves_second_relevant_doc(self):
        chunks = [
            Chunk(
                "c1",
                "doc-a",
                "financial_reports",
                1,
                "主营业务情况",
                "",
                "比亚迪 2025 年年度报告 研发投入占营业收入比例上升。",
                [],
                ["2025 年"],
                [],
                {"title": "annual_byd_2025_report", "extra_index_fields": ["比亚迪", "研发投入", "营业收入"]},
            ),
            Chunk(
                "c2",
                "doc-b",
                "financial_reports",
                1,
                "股东回报",
                "",
                "美的集团自 2019 年起连续实施股份回购方案。",
                [],
                ["2019 年"],
                [],
                {"title": "annual_midea_2024_report", "extra_index_fields": ["美的集团", "股份回购", "2019 年"]},
            ),
            Chunk(
                "c3",
                "doc-noise",
                "financial_reports",
                1,
                "经营情况讨论",
                "",
                "中国移动 2025 年营业收入增长。",
                [],
                ["2025 年"],
                [],
                {"title": "annual_chinamobile_2025_report", "extra_index_fields": ["中国移动", "营业收入"]},
            ),
        ]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")
        question = Question(
            qid="q3",
            domain="financial_reports",
            split="A",
            question="在提供的文档中，比亚迪 2025 年的研发投入占营业收入比例相较于 2024 年有所上升，且美的集团自 2019 年起连续实施了股份回购方案。",
            options={"A": "正确", "B": "错误"},
            answer_format="mcq",
            doc_ids=["doc-a", "doc-b"],
        )

        results = retrieve_with_variant(index, question, "broad_sparse_structured_rerank", top_k=3)

        self.assertTrue(results)
        retrieved_doc_ids = [result.doc_id for result in results]
        self.assertIn("doc-a", retrieved_doc_ids)
        self.assertIn("doc-b", retrieved_doc_ids)
        self.assertTrue(any("rerank_score" in result.metadata for result in results))


if __name__ == "__main__":
    unittest.main()
