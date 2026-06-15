"""BM25 索引构建辅助函数。"""

from __future__ import annotations

from pathlib import Path

from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.io.jsonl import read_jsonl
from agent.schemas import Chunk


def load_chunks(path: Path) -> list[Chunk]:
    """从 chunks.jsonl 加载 chunk 对象。"""
    return [Chunk.from_dict(row) for row in read_jsonl(path)]


def build_index_from_chunks(chunks_path: Path, index_path: Path, tokenizer_mode: str = "mixed") -> BM25SearchIndex:
    """读取 chunks 并构建/保存 BM25 索引。"""
    chunks = load_chunks(chunks_path)
    index = BM25SearchIndex.build(chunks, tokenizer_mode=tokenizer_mode)
    index.save(index_path)
    return index


def build_document_index_from_chunks(
    chunks_path: Path,
    index_path: Path,
    tokenizer_mode: str = "mixed",
) -> DocumentSearchIndex:
    """读取 chunks 并构建/保存文档级盲搜索引。"""
    chunks = load_chunks(chunks_path)
    index = DocumentSearchIndex.build(chunks, tokenizer_mode=tokenizer_mode)
    index.save(index_path)
    return index
