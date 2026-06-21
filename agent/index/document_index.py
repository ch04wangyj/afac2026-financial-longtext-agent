"""文档级 BM25 盲搜索引。

B 榜题目不提供 doc_ids，因此需要先从全库选择候选文档，再进入 chunk 级检索。
这里仍然只使用词法 BM25，不使用 embedding。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from agent.index.bm25 import BM25SearchIndex
from agent.schemas import Chunk


class DocumentSearchIndex:
    """把同一 doc_id 的 chunks 合并成文档级伪 chunk 进行检索。"""

    def __init__(self, index: BM25SearchIndex) -> None:
        self.index = index

    @classmethod
    def build(cls, chunks: list[Chunk], tokenizer_mode: str = "mixed", max_doc_chars: int = 24000) -> "DocumentSearchIndex":
        """从 chunk 列表构建文档级索引，长文档截断以控制索引体积。"""
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk.doc_id].append(chunk)

        doc_chunks: list[Chunk] = []
        for doc_id, items in sorted(by_doc.items()):
            first = items[0]
            title = str(first.metadata.get("title", ""))
            parts = [title]
            for item in items:
                extra_index_fields = item.metadata.get("extra_index_fields", [])
                fields = [
                    item.section,
                    item.clause_id,
                    item.text,
                    " ".join(item.tables),
                    " ".join(str(x) for x in extra_index_fields if str(x).strip()),
                ]
                text = "\n".join(field for field in fields if field)
                if text:
                    parts.append(text)
                if sum(len(part) for part in parts) >= max_doc_chars:
                    break
            doc_chunks.append(
                Chunk(
                    chunk_id=f"doc::{doc_id}",
                    doc_id=doc_id,
                    domain=first.domain,
                    page=None,
                    section="document",
                    clause_id="",
                    text="\n".join(parts)[:max_doc_chars],
                    metadata={"title": title, "doc_level": True},
                )
            )
        return cls(BM25SearchIndex.build(doc_chunks, tokenizer_mode=tokenizer_mode))

    def search_doc_ids(self, query: str, top_k: int = 8, domain: str | None = None) -> list[str]:
        """返回文档级候选 doc_id，可选按领域过滤。"""
        filter_doc_ids = None
        if domain:
            filter_doc_ids = {chunk.doc_id for chunk in self.index.chunks if chunk.domain == domain}
        results = self.index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source="document_bm25")
        return [result.doc_id for result in results]

    def save(self, path: Path) -> None:
        """保存文档级索引。"""
        self.index.save(path)

    @classmethod
    def load(cls, path: Path) -> "DocumentSearchIndex":
        """加载文档级索引。"""
        return cls(BM25SearchIndex.load(path))
