"""领域清洗与索引增强测试。"""

import unittest

from agent.index.document_index import DocumentSearchIndex
from agent.index.tokenizer import tokenize_chunk
from agent.preprocess.chunkers import chunk_document
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.extractors import PageText
from agent.schemas import Chunk, Document


class DomainIndexingTest(unittest.TestCase):
    def test_build_extra_index_fields_for_financial_reports_extracts_metric_keywords(self):
        chunk = {
            "domain": "financial_reports",
            "text": "营业收入同比增长8.2%，归属于上市公司股东的净利润增长6.1%。",
            "metadata": {},
        }
        fields = build_extra_index_fields(chunk)
        self.assertIn("营业收入", fields)
        self.assertIn("净利润", " ".join(fields))

    def test_build_extra_index_fields_for_insurance_includes_clause_id(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            domain="insurance",
            page=1,
            section="保险责任",
            clause_id="第一条",
            text="保险责任包括疾病身故给付。",
            metadata={},
        )
        fields = build_extra_index_fields(chunk)
        self.assertIn("第一条", fields)
        self.assertIn("保险责任", fields)

    def test_build_extra_index_fields_includes_structure_captions_and_parser_name(self):
        chunk = Chunk(
            chunk_id="c-table",
            doc_id="d1",
            domain="financial_reports",
            page=2,
            section="table",
            clause_id="",
            text="项目 | 2024年\n营业收入 | 10亿元",
            metadata={
                "chunk_type": "table",
                "caption": "主要会计数据",
                "parser_name": "docling",
            },
        )
        fields = build_extra_index_fields(chunk)
        self.assertIn("主要会计数据", fields)
        self.assertIn("docling", fields)

    def test_tokenize_chunk_uses_extra_index_fields(self):
        chunk = Chunk(
            chunk_id="c1",
            doc_id="d1",
            domain="financial_reports",
            page=1,
            section="",
            clause_id="",
            text="同比增长8.2%。",
            metadata={"extra_index_fields": ["营业收入"]},
        )
        tokens = tokenize_chunk(chunk, mode="mixed")
        joined = " ".join(tokens)
        self.assertIn("营业收入", joined)

    def test_document_index_search_doc_ids_uses_extra_index_fields(self):
        chunks = [
            Chunk(
                chunk_id="c1",
                doc_id="doc1",
                domain="financial_reports",
                page=1,
                section="",
                clause_id="",
                text="同比增长8.2%。",
                metadata={"title": "示例年报", "extra_index_fields": ["营业收入"]},
            ),
            Chunk(
                chunk_id="c2",
                doc_id="doc2",
                domain="financial_reports",
                page=1,
                section="",
                clause_id="",
                text="资产减值准备说明。",
                metadata={"title": "另一份年报"},
            ),
        ]
        index = DocumentSearchIndex.build(chunks, tokenizer_mode="mixed")
        results = index.search_doc_ids("营业收入", top_k=3, domain="financial_reports")
        self.assertTrue(results)
        self.assertEqual(results[0], "doc1")

    def test_chunk_document_emits_table_and_figure_chunks(self):
        doc = Document(doc_id="r1", domain="financial_reports", title="示例年报", path="dummy.pdf")
        pages = [
            PageText(
                page=15,
                text="营业收入同比增长。",
                tables=[{"text": "项目 | 2024年\n营业收入 | 10亿元", "caption": "主要会计数据"}],
                figures=[{"text": "收入趋势", "caption": "收入趋势图"}],
                parser_name="docling",
                metadata={},
            )
        ]
        chunks = chunk_document(doc, pages)
        kinds = [chunk.metadata.get("chunk_type", "text") for chunk in chunks]
        self.assertIn("table", kinds)
        self.assertIn("figure", kinds)


if __name__ == "__main__":
    unittest.main()
