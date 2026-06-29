"""脚本 15：从既有解析结果构建 V3 层级子块与 BM25F 索引。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.index.bm25 import BM25SearchIndex
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.preprocess.hierarchical_chunking import HierarchicalChunkConfig, build_hierarchical_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 V3 原子子块和层级 BM25 索引。")
    parser.add_argument("--chunks", type=Path, default=None, help="V2 或基础 chunks JSONL。")
    parser.add_argument("--children", type=Path, default=None, help="Output searchable child chunks JSONL.")
    parser.add_argument("--parents", type=Path, default=None, help="Output parent context chunks JSONL.")
    parser.add_argument("--index", type=Path, default=None, help="Output BM25 index pickle.")
    parser.add_argument("--target-chars", type=int, default=360)
    parser.add_argument("--max-chars", type=int, default=520)
    parser.add_argument("--min-chars", type=int, default=48)
    parser.add_argument("--tokenizer-mode", choices=["mixed", "char", "word"], default="mixed")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    source = args.chunks or _default_source(settings.processed_dir)
    children_path = args.children or settings.processed_dir / "chunks_v3_atomic.jsonl"
    parents_path = args.parents or settings.processed_dir / "chunks_v3_parents.jsonl"
    index_path = args.index or settings.processed_dir / "v3_atomic" / "bm25_index.pkl"

    rows = list(read_jsonl(source))
    parents, children = build_hierarchical_corpus(
        rows,
        HierarchicalChunkConfig(
            target_chars=args.target_chars,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        ),
    )
    write_jsonl(parents_path, (chunk.to_dict() for chunk in parents))
    write_jsonl(children_path, (chunk.to_dict() for chunk in children))

    index = BM25SearchIndex.build(
        children,
        tokenizer_mode=args.tokenizer_mode,
        parent_chunks=parents,
    )
    index.save(index_path)
    print(
        f"built V3 hierarchy: parents={len(parents)} children={len(children)} "
        f"source={source} index={index_path}"
    )


def _default_source(processed_dir: Path) -> Path:
    """优先复用 V2 已补充财务指标行的最终语料。"""
    candidates = [
        processed_dir / "chunks_financial_rows_final.jsonl",
        processed_dir / "chunks_financial_rows.jsonl",
        processed_dir / "chunks.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[-1]


if __name__ == "__main__":
    main()
