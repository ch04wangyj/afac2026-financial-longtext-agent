"""RAG 检索变体的单元测试。"""

import unittest

from agent.index.bm25 import BM25SearchIndex
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


if __name__ == "__main__":
    unittest.main()
