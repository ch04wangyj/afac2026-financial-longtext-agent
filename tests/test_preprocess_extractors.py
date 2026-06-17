"""预处理解析器路由测试。"""

import unittest
from pathlib import Path

from agent.preprocess import extractors
from agent.preprocess.extractors import PageText, choose_pdf_parser


class PreprocessExtractorsTest(unittest.TestCase):
    def test_choose_pdf_parser_uses_docling_for_all_pdf_domains(self):
        self.assertEqual(choose_pdf_parser("insurance"), "docling")
        self.assertEqual(choose_pdf_parser("financial_contracts"), "docling")
        self.assertEqual(choose_pdf_parser("financial_reports"), "docling")
        self.assertEqual(choose_pdf_parser("research"), "docling")
        self.assertEqual(choose_pdf_parser("regulatory"), "docling")


def test_read_pdf_uses_docling_adapter_parse_function(monkeypatch):
    called = {}

    def fake_parse_pdf_with_docling(path):
        called["path"] = path
        return [PageText(page=1, text="正文", tables=[], parser_name="docling")]

    monkeypatch.setattr("agent.preprocess.docling_adapter.parse_pdf_with_docling", fake_parse_pdf_with_docling)

    pages = extractors._read_pdf(Path("dummy.pdf"), "financial_reports")

    assert called["path"] == Path("dummy.pdf")
    assert len(pages) == 1
    assert pages[0].parser_name == "docling"


if __name__ == "__main__":
    unittest.main()
