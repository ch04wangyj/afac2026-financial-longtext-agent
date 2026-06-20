"""Logic/Linear/Graph/CRAG lite 查询构造与检索测试。"""

import unittest

from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.structured_queries import (
    build_graph_lite_queries,
    build_linear_entity_queries,
    build_logic_queries,
    extract_query_entities,
)
from agent.retrieve.variants import retrieve_with_variant
from agent.schemas import Chunk, Question


class StructuredRagVariantsTest(unittest.TestCase):
    def setUp(self):
        self.question = Question(
            qid="q1",
            domain="regulation",
            split="A",
            question="根据《客户尽职调查办法》，2024年银行识别受益所有人的要求是什么？",
            options={
                "A": "识别受益所有人并保存客户身份资料",
                "B": "不需要识别受益所有人",
            },
            answer_format="mcq",
            doc_ids=["doc1"],
        )

    def test_extract_query_entities_keeps_law_year_and_metric(self):
        entities = extract_query_entities(self.question.question)
        self.assertIn("《客户尽职调查办法》", entities)
        self.assertIn("2024年", entities)
        self.assertIn("受益所有人", entities)

    def test_extract_query_entities_keeps_company_and_product_signals(self):
        text = "比亚迪 2025 年年度报告显示研发投入增长，平安e生保等待期为 30 日。"
        entities = extract_query_entities(text)
        self.assertIn("比亚迪", entities)
        self.assertIn("2025 年", entities)
        self.assertIn("研发投入", entities)
        self.assertTrue(any("平安e生保" in item for item in entities))

    def test_structured_query_builders_return_deduped_queries(self):
        for builder in (build_logic_queries, build_linear_entity_queries, build_graph_lite_queries):
            queries = builder(self.question)
            self.assertTrue(queries)
            self.assertEqual(len(queries), len(set(queries)))

    def test_new_variants_retrieve_matching_document(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                domain="regulation",
                page=1,
                section="",
                clause_id="",
                text="《客户尽职调查办法》 银行 应当 识别 受益所有人 保存客户身份资料 2024年",
                tables=[],
                numbers=[],
                dates=["2024年"],
            ),
            Chunk(
                chunk_id="c2",
                doc_id="doc2",
                domain="regulation",
                page=1,
                section="",
                clause_id="",
                text="证券发行承销管理办法 询价 配售",
                tables=[],
                numbers=[],
                dates=[],
            ),
        ]
        index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")
        for variant in ("logic_lite_rrf", "linear_entity_rrf", "graph_lite_rrf", "crag_lite"):
            results = retrieve_with_variant(index, self.question, variant, top_k=2)
            self.assertTrue(results, variant)
            self.assertEqual(results[0].doc_id, "doc1", variant)


if __name__ == "__main__":
    unittest.main()
