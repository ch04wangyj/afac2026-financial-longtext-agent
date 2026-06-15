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
from agent.io.jsonl import write_jsonl
from agent.preprocess.pipeline import parse_document


def main() -> None:
    """读取 A 组题目引用的文档，完成解析和分块。"""
    parser = argparse.ArgumentParser(description="Parse raw documents and build documents/chunks JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents for smoke tests.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domains to include.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = load_questions(settings.questions_root, domains=args.domains)
    registry = DocRegistry(settings.raw_root)
    documents = registry.build_documents_for_questions(questions)
    if args.limit:
        documents = documents[: args.limit]

    parsed_docs = []
    all_chunks = []
    for idx, document in enumerate(documents, start=1):
        # 按题目引用文档逐个解析，方便失败时定位到具体领域和 doc_id。
        print(f"[{idx}/{len(documents)}] parsing {document.domain}/{document.doc_id}")
        parsed, chunks = parse_document(document)
        parsed_docs.append(parsed.to_dict())
        all_chunks.extend(chunk.to_dict() for chunk in chunks)

    write_jsonl(settings.processed_dir / "documents.jsonl", parsed_docs)
    write_jsonl(settings.processed_dir / "chunks.jsonl", all_chunks)
    print(f"wrote {len(parsed_docs)} documents and {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
