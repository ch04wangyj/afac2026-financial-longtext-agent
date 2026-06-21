"""预处理流水线与结构化 chunk 测试。"""

from agent.preprocess.chunkers import chunk_document
from agent.preprocess.extractors import PageText
from agent.schemas import Document


def test_chunk_document_emits_table_and_figure_chunks():
    doc = Document(doc_id="r1", domain="financial_reports", title="示例年报", path="dummy.pdf")
    pages = [
        PageText(
            page=15,
            text="营业收入同比增长。",
            tables=[{"text": "项目 | 2024年\n营业收入 | 10亿元", "caption": "主要会计数据"}],
            figures=[{"text": "收入趋势", "caption": "收入趋势图"}],
            parser_name="docling",
            metadata={},
        )
    ]
    chunks = chunk_document(doc, pages)
    kinds = [chunk.metadata.get("chunk_type", "text") for chunk in chunks]
    assert "table" in kinds
    assert "figure" in kinds
    table_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "table"]
    figure_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_type") == "figure"]
    assert table_chunks[0].metadata["caption"] == "主要会计数据"
    assert figure_chunks[0].metadata["caption"] == "收入趋势图"
    assert table_chunks[0].metadata["parser_name"] == "docling"
