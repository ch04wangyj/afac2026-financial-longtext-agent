"""预处理脚本流式写出测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from agent.schemas import Chunk, Document


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "01_prepare_docs.py"


def _load_prepare_docs_module():
    spec = importlib.util.spec_from_file_location("prepare_docs_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_process_documents_streaming_appends_results_per_document(tmp_path):
    module = _load_prepare_docs_module()
    settings = SimpleNamespace(processed_dir=tmp_path)
    documents = [
        Document(doc_id="doc-1", domain="insurance", title="Doc 1", path="a.pdf"),
        Document(doc_id="doc-2", domain="insurance", title="Doc 2", path="b.pdf"),
    ]

    appended_docs = []
    appended_chunk_batches = []

    def fake_parse_document(document):
        parsed = Document(
            doc_id=document.doc_id,
            domain=document.domain,
            title=document.title,
            path=document.path,
            raw_text=f"raw::{document.doc_id}",
            metadata={"page_count": 1},
        )
        chunks = [
            Chunk(
                chunk_id=f"{document.doc_id}-c1",
                doc_id=document.doc_id,
                domain=document.domain,
                page=1,
                section="",
                clause_id="",
                text=f"chunk::{document.doc_id}",
                metadata={},
            )
        ]
        return parsed, chunks

    def fake_append_jsonl(path, row):
        appended_docs.append((Path(path).name, row["doc_id"]))

    def fake_append_jsonl_rows(path, rows):
        rows = list(rows)
        appended_chunk_batches.append((Path(path).name, [row["chunk_id"] for row in rows]))
        return len(rows)

    doc_count, chunk_count = module.process_documents_streaming(
        documents,
        settings,
        parse_document_fn=fake_parse_document,
        append_row_fn=fake_append_jsonl,
        append_rows_fn=fake_append_jsonl_rows,
    )

    assert doc_count == 2
    assert chunk_count == 2
    assert appended_docs == [("documents.jsonl", "doc-1"), ("documents.jsonl", "doc-2")]
    assert appended_chunk_batches == [
        ("chunks.jsonl", ["doc-1-c1"]),
        ("chunks.jsonl", ["doc-2-c1"]),
    ]
