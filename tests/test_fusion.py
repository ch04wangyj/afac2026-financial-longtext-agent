"""RRF 排序融合的单元测试。"""

import unittest

from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import RetrievalResult


def result(chunk_id: str, score: float) -> RetrievalResult:
    """构造测试用检索结果。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id="doc",
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


if __name__ == "__main__":
    unittest.main()
