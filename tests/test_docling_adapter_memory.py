"""Docling 内存阶段与导出策略测试。"""

from __future__ import annotations

from pathlib import Path

from agent.preprocess import docling_adapter


def test_collect_docling_memory_profile_has_named_stages(monkeypatch):
    stages = []

    class FakeDoc:
        def export_to_dict(self):
            stages.append("export_to_dict")
            return {"pages": {"1": {"page_no": 1}}, "texts": [], "tables": [], "pictures": []}

        def export_to_markdown(self):
            stages.append("export_to_markdown")
            return "正文"

    class FakeResult:
        document = FakeDoc()

    class FakeConverter:
        def convert(self, path):
            stages.append("convert")
            return FakeResult()

    monkeypatch.setattr(docling_adapter, "build_docling_converter", lambda: FakeConverter())

    profile = docling_adapter.collect_docling_memory_profile(Path("dummy.pdf"))

    assert [row["stage"] for row in profile] == [
        "before_convert",
        "after_convert",
        "after_export_dict",
        "after_export_markdown",
        "after_pages_from_doc_dict",
    ]
    assert stages == ["convert", "export_to_dict", "export_to_markdown"]


def test_parse_pdf_with_docling_avoids_markdown_export_when_pages_exist(monkeypatch):
    called = {"markdown": 0}

    class FakeDoc:
        def export_to_dict(self):
            return {
                "pages": {"1": {"page_no": 1}},
                "texts": [{"text": "正文", "prov": [{"page_no": 1}]}],
                "tables": [],
                "pictures": [],
            }

        def export_to_markdown(self):
            called["markdown"] += 1
            return "正文"

    monkeypatch.setattr(docling_adapter, "_convert_with_docling", lambda path: (None, FakeDoc()))

    pages = docling_adapter.parse_pdf_with_docling(Path("dummy.pdf"))

    assert len(pages) == 1
    assert pages[0].text == "正文"
    assert called["markdown"] == 0
