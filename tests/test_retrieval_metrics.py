"""检索代理指标的单元测试。"""

import unittest

from agent.retrieve.metrics import evaluate_retrieval, summarize_metrics
from agent.schemas import Question, RetrievalResult


def rr(doc_id: str, chunk_id: str) -> RetrievalResult:
    """构造测试用 RetrievalResult。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="research",
        score=1.0,
        source="test",
        query="q",
        evidence_text="x",
    )


class RetrievalMetricsTest(unittest.TestCase):
    def test_doc_hit_metrics(self):
        question = Question(
            qid="q1",
            domain="research",
            split="A",
            question="q",
            options={"A": "a"},
            answer_format="mcq",
            doc_ids=["d2", "d3"],
        )
        metrics = evaluate_retrieval(question, [rr("d1", "c1"), rr("d2", "c2"), rr("d3", "c3")], "v", "mixed")
        self.assertFalse(metrics.hit_at_1)
        self.assertTrue(metrics.hit_at_3)
        self.assertTrue(metrics.all_gold_at_10)
        self.assertAlmostEqual(metrics.recall_at_10, 1.0)
        self.assertAlmostEqual(metrics.mrr_at_10, 0.5)

    def test_summary(self):
        question = Question("q1", "research", "A", "q", {"A": "a"}, "mcq", doc_ids=["d2"])
        metrics = [evaluate_retrieval(question, [rr("d2", "c2")], "v", "mixed")]
        summary = summarize_metrics(metrics)
        all_row = [row for row in summary if row["domain"] == "ALL"][0]
        self.assertEqual(all_row["questions"], 1)
        self.assertEqual(all_row["hit_at_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
