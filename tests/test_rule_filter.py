"""规则证据压缩的单元测试。"""

import unittest

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.schemas import Question, RetrievalResult


def rr(chunk_id: str, doc_id: str, score: float, text: str) -> RetrievalResult:
    """构造测试用 RetrievalResult。"""
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_contracts",
        score=score,
        source="test",
        query="",
        evidence_text=text,
    )


class RuleEvidenceCompressorTest(unittest.TestCase):
    def test_multi_doc_question_keeps_each_gold_doc(self):
        question = Question(
            qid="q1",
            domain="financial_contracts",
            split="A",
            question="比较两份文档的发行人和受托管理人",
            options={"A": "第一份文档发行人为甲", "B": "第二份文档受托管理人为乙"},
            answer_format="multi",
            doc_ids=["doc1", "doc2"],
        )
        results = [
            rr("c1", "doc1", 10.0, "第一份文档 发行人 甲"),
            rr("c2", "doc1", 9.0, "第一份文档 主体评级 AAA"),
            rr("c3", "doc2", 1.0, "第二份文档 受托管理人 乙"),
        ]
        evidence = RuleEvidenceCompressor(top_k=2).compress(question, results)
        self.assertEqual({item.doc_id for item in evidence}, {"doc1", "doc2"})


if __name__ == "__main__":
    unittest.main()
