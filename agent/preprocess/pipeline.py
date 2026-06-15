"""预处理流水线：Document -> 页面文本 -> chunks。"""

from __future__ import annotations

from agent.preprocess.chunkers import chunk_document
from agent.preprocess.extractors import extract_pages
from agent.schemas import Chunk, Document


def parse_document(document: Document) -> tuple[Document, list[Chunk]]:
    """解析单个文档，并返回带 raw_text 的 Document 与 chunk 列表。"""
    pages = extract_pages(document.path_obj, document.domain)
    raw_text = "\n\n".join(page.text for page in pages if page.text)
    parsed_doc = Document(
        doc_id=document.doc_id,
        domain=document.domain,
        title=document.title,
        path=document.path,
        raw_text=raw_text,
        metadata={**document.metadata, "page_count": len(pages)},
    )
    chunks = chunk_document(parsed_doc, pages)
    return parsed_doc, chunks
