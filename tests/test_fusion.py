"""RRF 排序融合的单元测试。"""

import unittest

from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import RetrievalResult


def result(chunk_id: str, score: float, doc_id: str = "doc") -> RetrievalResult:
    """构造测试用检索结果。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="insurance",
        score=score,
        source="test",
        query="q",
        evidence_text="text",
    )


class FusionTest(unittest.TestCase):
    def test_rrf_promotes_repeated_chunks(self):
        fused = reciprocal_rank_fusion(
            [
                [result("a", 3.0), result("b", 2.0)],
                [result("b", 3.0), result("c", 2.0)],
            ],
            top_k=3,
        )
        self.assertEqual(fused[0].chunk_id, "b")
        self.assertEqual({item.chunk_id for item in fused}, {"a", "b", "c"})

    def test_rrf_rescues_document_repeated_across_queries_with_different_chunks(self):
        fused = reciprocal_rank_fusion(
            [
                [result("a1", 5.0, "doc-a"), result("target-1", 1.0, "target")],
                [result("b1", 5.0, "doc-b"), result("target-2", 1.0, "target")],
            ],
            top_k=2,
            doc_rescue_top_n=2,
        )

        self.assertIn("target", {item.doc_id for item in fused})
        self.assertTrue(any(item.source.startswith("rrf_doc_rescue:") for item in fused))

    def test_weighted_rrf_promotes_precise_metric_query(self):
        broad = [result("noise", 10.0), result("target", 9.0)]
        metric = [result("target", 10.0), result("noise", 9.0)]

        fused = reciprocal_rank_fusion([broad, metric], top_k=2, weights=[0.5, 3.0])

        self.assertEqual(fused[0].chunk_id, "target")

    def test_weighted_rrf_rejects_mismatched_weight_count(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[result("a", 1.0)]], weights=[])


if __name__ == "__main__":
    unittest.main()
