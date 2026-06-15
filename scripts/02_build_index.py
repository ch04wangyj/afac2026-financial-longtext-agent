"""脚本 02：从 chunks.jsonl 构建 BM25 检索索引。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.index.build import build_document_index_from_chunks, build_index_from_chunks


def main() -> None:
    """按指定 tokenizer 模式构建可持久化词法索引。"""
    parser = argparse.ArgumentParser(description="Build lexical BM25 index from chunks.jsonl.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--doc-index", type=Path, default=None)
    parser.add_argument("--skip-doc-index", action="store_true", help="Only build chunk-level index.")
    parser.add_argument("--tokenizer-mode", choices=["mixed", "char", "word"], default="mixed")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    chunks_path = args.chunks or settings.processed_dir / "chunks.jsonl"
    index_path = args.index or settings.index_dir / "bm25_index.pkl"
    index = build_index_from_chunks(chunks_path, index_path, tokenizer_mode=args.tokenizer_mode)
    print(f"indexed {len(index.chunks)} chunks -> {index_path}")
    if not args.skip_doc_index:
        doc_index_path = args.doc_index or settings.index_dir / "document_bm25_index.pkl"
        doc_index = build_document_index_from_chunks(chunks_path, doc_index_path, tokenizer_mode=args.tokenizer_mode)
        print(f"indexed {len(doc_index.index.chunks)} documents -> {doc_index_path}")


if __name__ == "__main__":
    main()
