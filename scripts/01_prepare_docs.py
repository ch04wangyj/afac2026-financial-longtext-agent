"""脚本 01：解析原始文档并生成 documents.jsonl / chunks.jsonl。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.doc_registry import DocRegistry
from agent.data.questions import load_questions
from agent.io.jsonl import append_jsonl, append_jsonl_rows
from agent.preprocess.pipeline import parse_document
from agent.runtime.process_guard import assert_no_other_heavy_python_jobs
from agent.schemas import Document


def _reset_output_paths(settings: Settings) -> tuple[Path, Path]:
    documents_path = settings.processed_dir / "documents.jsonl"
    chunks_path = settings.processed_dir / "chunks.jsonl"
    for path in (documents_path, chunks_path):
        if path.exists():
            path.unlink()
    return documents_path, chunks_path


def process_documents_streaming(
    documents: list[Document],
    settings: Settings,
    *,
    parse_document_fn=parse_document,
    append_row_fn=append_jsonl,
    append_rows_fn=append_jsonl_rows,
) -> tuple[int, int]:
    documents_path, chunks_path = _reset_output_paths(settings)
    doc_count = 0
    chunk_count = 0
    for idx, document in enumerate(documents, start=1):
        print(f"[{idx}/{len(documents)}] parsing {document.domain}/{document.doc_id}")
        parsed, chunks = parse_document_fn(document)
        append_row_fn(documents_path, parsed.to_dict())
        doc_count += 1
        chunk_count += append_rows_fn(chunks_path, (chunk.to_dict() for chunk in chunks))
    return doc_count, chunk_count


def main() -> None:
    """读取 A 组题目引用的文档，完成解析和分块。"""
    parser = argparse.ArgumentParser(description="Parse raw documents and build documents/chunks JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents for smoke tests.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domains to include.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    assert_no_other_heavy_python_jobs()
    questions = load_questions(settings.questions_root, domains=args.domains)
    registry = DocRegistry(settings.raw_root)
    documents = registry.build_documents_for_questions(questions)
    if args.limit:
        documents = documents[: args.limit]

    doc_count, chunk_count = process_documents_streaming(documents, settings)
    print(f"wrote {doc_count} documents and {chunk_count} chunks")


if __name__ == "__main__":
    main()
