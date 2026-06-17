"""Docling 适配层测试。"""

import unittest

from agent.preprocess import docling_adapter
from agent.preprocess.docling_adapter import ParsedPage, _pages_from_doc_dict


class DoclingAdapterTest(unittest.TestCase):
    def test_parsed_page_keeps_text_tables_figures_and_metadata(self):
        page = ParsedPage(
            page=2,
            text="营业收入同比增长。",
            tables=["项目 | 金额"],
            figures=[{"label": "chart", "caption": "收入趋势图"}],
            parser_name="docling",
            ocr_used=False,
        )

        self.assertEqual(page.page, 2)
        self.assertEqual(page.tables, ["项目 | 金额"])
        self.assertEqual(page.figures[0]["label"], "chart")
        self.assertEqual(page.parser_name, "docling")

    def test_pages_from_doc_dict_groups_text_by_page_number(self):
        doc_dict = {
            "pages": {
                "1": {"page_no": 1, "size": {}},
                "2": {"page_no": 2, "size": {}},
            },
            "texts": [
                {"label": "paragraph", "text": "第一页正文", "prov": [{"page_no": 1}]},
                {"label": "paragraph", "text": "第二页正文", "prov": [{"page_no": 2}]},
            ],
            "tables": [],
            "pictures": [],
        }

        pages = _pages_from_doc_dict(doc_dict)

        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0].page, 1)
        self.assertEqual(pages[0].text, "第一页正文")
        self.assertEqual(pages[1].page, 2)
        self.assertEqual(pages[1].text, "第二页正文")

    def test_pages_from_doc_dict_preserves_table_and_figure_caption(self):
        doc_dict = {
            "pages": {"1": {"page_no": 1}},
            "texts": [{"label": "paragraph", "text": "正文", "prov": [{"page_no": 1}]}],
            "tables": [
                {
                    "label": "table",
                    "captions": [{"text": "主要会计数据"}],
                    "prov": [{"page_no": 1}],
                    "data": {"rows": [["项目", "2024年"], ["营业收入", "10亿元"]]},
                }
            ],
            "pictures": [
                {
                    "label": "picture",
                    "captions": [{"text": "收入趋势图"}],
                    "prov": [{"page_no": 1}],
                    "text": "图示说明",
                }
            ],
        }

        pages = _pages_from_doc_dict(doc_dict)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].metadata["table_count"], 1)
        self.assertEqual(pages[0].metadata["figure_count"], 1)
        self.assertEqual(pages[0].metadata["table_captions"], ["主要会计数据"])
        self.assertEqual(pages[0].figures[0]["caption"], "收入趋势图")


def test_build_docling_converter_uses_stable_pdf_pipeline(monkeypatch):
    captured = {}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            captured["pipeline_kwargs"] = kwargs

    class FakePdfFormatOption:
        def __init__(self, *, pipeline_options, backend):
            captured["pipeline_options"] = pipeline_options
            captured["backend"] = backend

    class FakeConverter:
        def __init__(self, *, format_options):
            captured["format_options"] = format_options

    class FakeInputFormat:
        PDF = "pdf"

    class FakeBackend:
        pass

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeConverter, raising=False)
    monkeypatch.setattr(docling_adapter, "PdfFormatOption", FakePdfFormatOption, raising=False)
    monkeypatch.setattr(docling_adapter, "PdfPipelineOptions", FakePdfPipelineOptions, raising=False)
    monkeypatch.setattr(docling_adapter, "InputFormat", FakeInputFormat, raising=False)
    monkeypatch.setattr(docling_adapter, "PyPdfiumDocumentBackend", FakeBackend, raising=False)
    monkeypatch.setattr(docling_adapter, "_DOC_CONVERTER", None, raising=False)

    converter = docling_adapter.build_docling_converter()

    assert isinstance(converter, FakeConverter)
    assert captured["pipeline_kwargs"] == {
        "do_ocr": False,
        "do_table_structure": True,
        "force_backend_text": True,
        "images_scale": 0.5,
    }
    assert captured["backend"] is FakeBackend
    assert "pdf" in captured["format_options"]


def test_build_docling_converter_reuses_singleton(monkeypatch):
    calls = {"count": 0}

    class FakePdfPipelineOptions:
        def __init__(self, **kwargs):
            pass

    class FakePdfFormatOption:
        def __init__(self, *, pipeline_options, backend):
            pass

    class FakeConverter:
        def __init__(self, *, format_options):
            calls["count"] += 1
            self.format_options = format_options

    class FakeInputFormat:
        PDF = "pdf"

    class FakeBackend:
        pass

    monkeypatch.setattr(docling_adapter, "DocumentConverter", FakeConverter, raising=False)
    monkeypatch.setattr(docling_adapter, "PdfFormatOption", FakePdfFormatOption, raising=False)
    monkeypatch.setattr(docling_adapter, "PdfPipelineOptions", FakePdfPipelineOptions, raising=False)
    monkeypatch.setattr(docling_adapter, "InputFormat", FakeInputFormat, raising=False)
    monkeypatch.setattr(docling_adapter, "PyPdfiumDocumentBackend", FakeBackend, raising=False)
    monkeypatch.setattr(docling_adapter, "_DOC_CONVERTER", None, raising=False)

    first = docling_adapter.build_docling_converter()
    second = docling_adapter.build_docling_converter()

    assert first is second
    assert calls["count"] == 1


if __name__ == "__main__":
    unittest.main()
