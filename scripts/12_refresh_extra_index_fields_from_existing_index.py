from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.index.bm25 import BM25SearchIndex
from agent.io.jsonl import append_jsonl_rows
from agent.preprocess.domain_indexing import build_extra_index_fields


def main() -> None:
    settings = Settings.from_env()
    index_path = settings.index_dir / "bm25_index.pkl"
    chunks_path = settings.processed_dir / "chunks.jsonl"
    index = BM25SearchIndex.load(index_path)
    chunks = index.chunks
    if chunks_path.exists():
        chunks_path.unlink()
    for chunk in chunks:
        chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
    written = append_jsonl_rows(chunks_path, (chunk.to_dict() for chunk in chunks))
    print(f"rewrote {written} chunks -> {chunks_path}")


if __name__ == "__main__":
    main()
