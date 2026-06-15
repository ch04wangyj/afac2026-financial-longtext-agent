"""文档级盲搜索引的单元测试。"""

import unittest

from agent.index.document_index import DocumentSearchIndex
from agent.schemas import Chunk


class DocumentSearchIndexTest(unittest.TestCase):
    def test_search_doc_ids_by_domain(self):
        chunks = [
            Chunk("c1", "doc1", "financial_contracts", 1, "", "", "广东省广晟控股集团 发行人", [], [], []),
            Chunk("c2", "doc2", "financial_contracts", 1, "", "", "国信证券 受托管理人", [], [], []),
            Chunk("c3", "doc3", "research", 1, "", "", "行业研报 国信证券", [], [], []),
        ]
        index = DocumentSearchIndex.build(chunks, tokenizer_mode="mixed")
        results = index.search_doc_ids("国信证券 受托管理人", top_k=3, domain="financial_contracts")
        self.assertTrue(results)
        self.assertEqual(results[0], "doc2")
        self.assertNotIn("doc3", results)


if __name__ == "__main__":
    unittest.main()
