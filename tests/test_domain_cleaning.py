"""领域样本文本分析与规则草案生成测试。"""

import unittest

from agent.preprocess.chunkers import chunk_document
from agent.preprocess.domain_cleaning import clean_domain_text
from agent.preprocess.domain_rules import infer_candidate_rules
from agent.preprocess.extractors import PageText
from agent.schemas import Document


class DomainRulesTest(unittest.TestCase):
    def test_infer_candidate_rules_flags_toc_noise_for_financial_reports(self):
        text = "目录\n第一节 公司简介\n第二节 会计数据和财务指标\n第三节 管理层讨论与分析"
        rules = infer_candidate_rules(domain="financial_reports", sample_text=text)
        self.assertIn("drop_toc_blocks", rules["cleaning_rules"])

    def test_clean_domain_text_drops_financial_report_toc_lines(self):
        raw = "目录\n第一节 公司简介\n第二节 会计数据和财务指标\n经营情况讨论"
        cleaned = clean_domain_text(
            domain="financial_reports",
            text=raw,
            rules=["drop_toc_blocks"],
        )
        self.assertNotIn("目录", cleaned)
        self.assertNotIn("第一节 公司简介", cleaned)
        self.assertIn("经营情况讨论", cleaned)

    def test_chunk_document_applies_financial_report_cleaning_rules(self):
        document = Document(
            doc_id="report1",
            domain="financial_reports",
            title="示例年报",
            path="dummy.pdf",
        )
        pages = [
            PageText(
                page=1,
                text="目录\n第一节 公司简介\n第二节 会计数据和财务指标\n\n营业收入同比增长8.2%",
                tables=[],
            )
        ]
        chunks = chunk_document(document, pages)
        self.assertTrue(chunks)
        merged = "\n".join(chunk.text for chunk in chunks)
        self.assertNotIn("目录", merged)
        self.assertNotIn("第一节 公司简介", merged)
        self.assertIn("营业收入同比增长8.2", merged)


if __name__ == "__main__":
    unittest.main()
