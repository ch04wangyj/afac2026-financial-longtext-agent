"""脚本 22：构建 V15 增量索引。

合并 V14 已验证语料 + V15 版面算法深化块 + Qwen-VL 离线提取表格块，
生成 V15 索引。不删除 V14 任何块，仅做增量补充和去重。

用法：
    python scripts/22_build_v15_index.py --strict
    python scripts/22_build_v15_index.py --strict --output-dir processed_data/v15_layout
"""

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
from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.layout_pdf import LayoutParseConfig, build_layout_supplement_chunks
from agent.preprocess.normalization import compact_for_search, normalize_text
from agent.schemas import Chunk, Document


DEFAULT_DOMAINS = ["financial_reports", "research"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V15 incremental index: V14 + VL tables + layout deepening.")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--v14-children", type=Path, default=None, help="V14 combined children JSONL")
    parser.add_argument("--v14-parents", type=Path, default=None, help="V14 parents JSONL")
    parser.add_argument("--vl-results", type=Path, default=None, help="Qwen-VL extraction results JSONL")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-mode", choices=["mixed", "char", "word"], default="mixed")
    parser.add_argument("--strict", action="store_true", help="任一 PDF 解析失败即停止。")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    output_dir = args.output_dir or settings.processed_dir / "v15_layout"
    output_dir.mkdir(parents=True, exist_ok=True)
    v14_children_path = args.v14_children or settings.processed_dir / "v14_layout" / "chunks_combined.jsonl"
    v14_parents_path = args.v14_parents or settings.processed_dir / "chunks_v13_parents.jsonl"
    vl_results_path = args.vl_results or settings.processed_dir / "v15_vl_tables" / "vl_table_results.jsonl"

    print("=" * 60)
    print("V15 Incremental Index Build")
    print("=" * 60)

    # 1. 加载 V14 已验证语料
    print("\n[1/5] Loading V14 combined children...")
    if not v14_children_path.exists():
        print(f"  ERROR: V14 children not found at {v14_children_path}")
        print(f"  Run scripts/19_build_layout_index.py first.")
        sys.exit(1)
    v14_children = [Chunk.from_dict(row) for row in read_jsonl(v14_children_path)]
    print(f"  V14 children: {len(v14_children)}")

    parents = [Chunk.from_dict(row) for row in read_jsonl(v14_parents_path)] if v14_parents_path.exists() else []
    print(f"  V14 parents: {len(parents)}")

    # 2. 加载 Qwen-VL 离线提取表格
    print("\n[2/5] Loading Qwen-VL extracted tables...")
    vl_chunks: list[Chunk] = []
    if vl_results_path.exists():
        for row in read_jsonl(vl_results_path):
            if not row.get("valid"):
                continue
            text = row.get("text", "")
            if not text or text == "NOT_A_TABLE":
                continue
            vl_chunk = _make_vl_chunk(row, text)
            vl_chunks.append(vl_chunk)
        print(f"  Qwen-VL valid table chunks: {len(vl_chunks)}")
    else:
        print(f"  No VL results found at {vl_results_path} (skipping)")

    # 3. 重建 V15 版面深化块（B1-B4 改进后的 layout_pdf）
    print("\n[3/5] Building V15 layout-deepened supplement chunks...")
    questions = load_questions(settings.questions_root, domains=args.domains)
    documents = DocRegistry(settings.raw_root).build_documents_for_questions(questions)
    documents = [document for document in documents if document.path_obj.suffix.lower() == ".pdf"]
    if args.limit:
        documents = documents[: args.limit]

    supplement: list[Chunk] = []
    failures: list[dict[str, str]] = []
    per_document: list[dict[str, object]] = []
    config = LayoutParseConfig()  # V15 默认启用 B1-B4
    for index, document in enumerate(documents, start=1):
        print(f"  [{index}/{len(documents)}] layout parsing {document.domain}/{document.doc_id}", flush=True)
        try:
            chunks = build_layout_supplement_chunks(document, config=config)
        except Exception as exc:
            failure = {"doc_id": document.doc_id, "domain": document.domain, "error": repr(exc)}
            failures.append(failure)
            print(f"    failed: {failure['error']}", flush=True)
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
        print(f"    emitted={len(chunks)} types={dict(counts)}", flush=True)

    print(f"  V15 layout supplement: {len(supplement)} chunks")
    if failures:
        print(f"  Failures: {len(failures)}")

    # 4. 合并：V14 + VL + V15 深化，去重
    print("\n[4/5] Merging V14 + VL + V15 supplement...")
    combined = merge_v15_chunks(v14_children, vl_chunks, supplement)
    print(f"  Combined children: {len(combined)}")
    print(f"    V14 base: {len(v14_children)}")
    print(f"    VL tables added: {len(vl_chunks)}")
    print(f"    V15 supplement added: {len(combined) - len(v14_children) - len(vl_chunks)}")

    # 5. 构建索引
    print("\n[5/5] Building BM25F index...")
    write_jsonl(output_dir / "chunks_vl.jsonl", (chunk.to_dict() for chunk in vl_chunks))
    write_jsonl(output_dir / "chunks_v15_supplement.jsonl", (chunk.to_dict() for chunk in supplement))
    write_jsonl(output_dir / "chunks_v15_combined.jsonl", (chunk.to_dict() for chunk in combined))
    index_path = output_dir / "bm25_index.pkl"
    BM25SearchIndex.build(combined, tokenizer_mode=args.tokenizer_mode, parent_chunks=parents).save(index_path)

    report = {
        "version": "v15",
        "domains": args.domains,
        "v14_base_children": len(v14_children),
        "vl_table_chunks": len(vl_chunks),
        "v15_supplement_chunks": len(supplement),
        "combined_children": len(combined),
        "failures": failures,
        "index": str(index_path),
    }
    write_json(output_dir / "build_report.json", report)
    print(f"\n{'=' * 60}")
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=2))
    print(f"{'=' * 60}")


def _make_vl_chunk(row: dict, text: str) -> Chunk:
    """把 Qwen-VL 提取结果转为索引可用的 Chunk。"""
    import hashlib

    doc_id = row.get("doc_id", "")
    domain = row.get("domain", "")
    page = row.get("page_index", 0) + 1
    file_name = row.get("file_name", "")
    digest = hashlib.sha1(f"vl:{doc_id}:{page}:{text[:100]}".encode("utf-8")).hexdigest()[:12]
    chunk = Chunk(
        chunk_id=f"vl:{digest}:p{page}:0",
        doc_id=doc_id,
        domain=domain,
        page=page,
        section="Qwen-VL图像表格提取",
        clause_id="",
        text=text,
        tables=[text[:200]],
        numbers=extract_numbers(text),
        dates=extract_dates(text),
        metadata={
            "title": file_name,
            "path": f"{domain}/{file_name}",
            "chunk_type": "vl_table_row",
            "hierarchy_level": "child",
            "parser_name": "qwen_vl_offline_v15",
            "vl_model": row.get("metadata", {}).get("vl_model", "qwen-vl-max"),
            "vl_page_index": row.get("page_index", 0),
            "vl_usage": row.get("usage", {}),
        },
    )
    chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
    return chunk


def merge_v15_chunks(
    base: list[Chunk],
    vl_chunks: list[Chunk],
    supplement: list[Chunk],
) -> list[Chunk]:
    """按文档和规范化正文去重，保留 V14 原块的稳定顺序。"""
    output = list(base)
    seen = {(chunk.doc_id, _compact(chunk.text)) for chunk in base}

    added_vl = 0
    for chunk in vl_chunks:
        key = (chunk.doc_id, _compact(chunk.text))
        if not key[1] or key in seen:
            continue
        output.append(chunk)
        seen.add(key)
        added_vl += 1

    added_supplement = 0
    for chunk in supplement:
        key = (chunk.doc_id, _compact(chunk.text))
        if not key[1] or key in seen:
            continue
        output.append(chunk)
        seen.add(key)
        added_supplement += 1

    return output


def _compact(value: str) -> str:
    return "".join(str(value or "").split()).casefold()


if __name__ == "__main__":
    main()
