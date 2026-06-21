"""脚本 13：基于现有 text chunks 生成财报指标行 chunks，无需重新解析 PDF。"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.io.jsonl import append_jsonl_rows, read_jsonl
from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.financial_rows import extract_financial_metric_rows, format_financial_metric_row
from agent.schemas import Chunk


def augment_financial_metric_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """保留现有非行级 chunks，并为财报 text chunk 追加结构化指标行。"""
    base_rows = [row for row in rows if (row.get("metadata") or {}).get("chunk_type") != "financial_metric_row"]
    generated: list[dict] = []
    for row in base_rows:
        if row.get("domain") != "financial_reports":
            continue
        metadata = dict(row.get("metadata") or {})
        if metadata.get("chunk_type", "text") != "text":
            continue
        title = str(metadata.get("title") or "")
        default_year_match = re.search(r"20\d{2}", title)
        default_year = default_year_match.group(0) if default_year_match else ""
        for row_index, financial_row in enumerate(
            extract_financial_metric_rows(str(row.get("text") or ""), default_year=default_year)
        ):
            body = format_financial_metric_row(financial_row, title=title)
            digest = hashlib.sha1(f"{row['chunk_id']}:{body}".encode("utf-8")).hexdigest()[:10]
            chunk = Chunk(
                chunk_id=f"{row['chunk_id']}:fr{row_index}:{digest}",
                doc_id=str(row["doc_id"]),
                domain="financial_reports",
                page=row.get("page"),
                section="financial_metric_row",
                clause_id="",
                text=body,
                tables=[financial_row["raw_row"]],
                numbers=extract_numbers(financial_row["raw_row"]),
                dates=extract_dates(financial_row["raw_row"]),
                metadata={
                    "title": metadata.get("title", ""),
                    "path": metadata.get("path", ""),
                    "parser_name": metadata.get("parser_name", ""),
                    "chunk_type": "financial_metric_row",
                    "parent_chunk_id": row["chunk_id"],
                    "financial_row": financial_row,
                },
            )
            chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
            generated.append(chunk.to_dict())
    return [*base_rows, *generated], len(generated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append deterministic financial metric row chunks.")
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    source = args.chunks or settings.processed_dir / "chunks.jsonl"
    output = args.output or settings.processed_dir / "chunks_financial_rows.jsonl"
    rows, generated_count = augment_financial_metric_rows(read_jsonl(source))
    if output.exists():
        output.unlink()
    append_jsonl_rows(output, rows)
    print(f"wrote {len(rows)} chunks ({generated_count} financial metric rows) -> {output}")


if __name__ == "__main__":
    main()
