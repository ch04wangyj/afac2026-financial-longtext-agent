"""脚本 19：构建 V4 PDF 版面/表格增量语料和 BM25F 索引。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.doc_registry import DocRegistry
from agent.data.questions import load_questions
from agent.index.bm25 import BM25SearchIndex
from agent.io.jsonl import read_jsonl, write_json, write_jsonl
from agent.preprocess.layout_pdf import LayoutParseConfig, build_layout_supplement_chunks
from agent.schemas import Chunk


DEFAULT_DOMAINS = ["financial_reports", "research"]


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 V4 确定性 PDF 版面增量索引。")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-children", type=Path, default=None)
    parser.add_argument("--base-parents", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-mode", choices=["mixed", "char", "word"], default="mixed")
    parser.add_argument("--strict", action="store_true", help="任一 PDF 解析失败即停止。")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    output_dir = args.output_dir or settings.processed_dir / "v4_layout"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_children_path = args.base_children or settings.processed_dir / "chunks_v3_atomic.jsonl"
    base_parents_path = args.base_parents or settings.processed_dir / "chunks_v3_parents.jsonl"

    questions = load_questions(settings.questions_root, domains=args.domains)
    documents = DocRegistry(settings.raw_root).build_documents_for_questions(questions)
    documents = [document for document in documents if document.path_obj.suffix.lower() == ".pdf"]
    if args.limit:
        documents = documents[: args.limit]

    supplement: list[Chunk] = []
    failures: list[dict[str, str]] = []
    per_document: list[dict[str, object]] = []
    config = LayoutParseConfig()
    for index, document in enumerate(documents, start=1):
        print(f"[{index}/{len(documents)}] layout parsing {document.domain}/{document.doc_id}", flush=True)
        try:
            chunks = build_layout_supplement_chunks(document, config=config)
        except Exception as exc:
            failure = {"doc_id": document.doc_id, "domain": document.domain, "error": repr(exc)}
            failures.append(failure)
            print(f"  failed: {failure['error']}", flush=True)
            if args.strict:
                raise
            continue
        supplement.extend(chunks)
        counts = Counter(str(chunk.metadata.get("chunk_type", "")) for chunk in chunks)
        per_document.append(
            {
                "doc_id": document.doc_id,
                "domain": document.domain,
                "chunks": len(chunks),
                "chunk_types": dict(counts),
            }
        )
        print(f"  emitted={len(chunks)} types={dict(counts)}", flush=True)

    base_children = [Chunk.from_dict(row) for row in read_jsonl(base_children_path)]
    parents = [Chunk.from_dict(row) for row in read_jsonl(base_parents_path)]
    combined = merge_supplement_chunks(base_children, supplement)

    write_jsonl(output_dir / "chunks_layout.jsonl", (chunk.to_dict() for chunk in supplement))
    write_jsonl(output_dir / "chunks_combined.jsonl", (chunk.to_dict() for chunk in combined))
    index_path = output_dir / "bm25_index.pkl"
    BM25SearchIndex.build(combined, tokenizer_mode=args.tokenizer_mode, parent_chunks=parents).save(index_path)
    report = {
        "domains": args.domains,
        "documents": len(documents),
        "base_children": len(base_children),
        "supplement_chunks": len(supplement),
        "combined_children": len(combined),
        "supplement_types": dict(Counter(str(chunk.metadata.get("chunk_type", "")) for chunk in supplement)),
        "failures": failures,
        "per_document": per_document,
        "index": str(index_path),
    }
    write_json(output_dir / "build_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_document"}, ensure_ascii=False))


def merge_supplement_chunks(base: list[Chunk], supplement: list[Chunk]) -> list[Chunk]:
    """按文档和规范化正文去重，保留 V3 原块的稳定顺序。"""
    output = list(base)
    seen = {(chunk.doc_id, _compact(chunk.text)) for chunk in base}
    for chunk in supplement:
        key = (chunk.doc_id, _compact(chunk.text))
        if not key[1] or key in seen:
            continue
        output.append(chunk)
        seen.add(key)
    return output


def _compact(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


if __name__ == "__main__":
    main()
