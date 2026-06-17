"""Docling 适配层：统一 Docling 输出，供预处理主线和样本导出复用。"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.preprocess.normalization import normalize_text

DocumentConverter = None
PdfFormatOption = None
InputFormat = None
PdfPipelineOptions = None
PyPdfiumDocumentBackend = None
_DOC_CONVERTER = None


@dataclass
class ParsedPage:
    """统一的页级结构，避免上层逻辑直接依赖 Docling 原始对象格式。"""

    page: int | None
    text: str
    tables: list[dict[str, Any] | str] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    parser_name: str = "docling"
    ocr_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoclingSampleBundle:
    """导出的样本产物及其结构摘要。"""

    markdown: str
    text: str
    tables: list[dict[str, Any]]
    figures: list[dict[str, Any]]
    pages: list[dict[str, Any]]


def parse_pdf_with_docling(path: Path) -> list[ParsedPage]:
    """用 Docling 解析 PDF，统一转换为 ParsedPage 列表。"""
    _, document = _convert_with_docling(path)
    doc_dict = _safe_export_dict(document)
    pages = _pages_from_doc_dict(doc_dict)
    if pages:
        return pages
    markdown = _safe_export_markdown(document)
    return [
        ParsedPage(
            page=None,
            text=normalize_text(markdown),
            parser_name="docling",
            metadata={"source": str(path), "fallback_full_document": True},
        )
    ]


def export_docling_sample_bundle(path: Path, output_dir: Path) -> DoclingSampleBundle:
    """把单份 Docling 样本导出成 markdown/text/summary 文件。"""
    _, document = _convert_with_docling(path)
    doc_dict = _safe_export_dict(document)
    pages = _pages_from_doc_dict(doc_dict)
    markdown = _safe_export_markdown(document)
    text = normalize_text(markdown)
    page_rows = [asdict(page) for page in pages]
    tables = _extract_tables(doc_dict)
    figures = _extract_figures(doc_dict)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full.md").write_text(markdown, encoding="utf-8")
    (output_dir / "full.txt").write_text(text, encoding="utf-8")
    (output_dir / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "figures.json").write_text(json.dumps(figures, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "pages.json").write_text(json.dumps(page_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return DoclingSampleBundle(
        markdown=markdown,
        text=text,
        tables=tables,
        figures=figures,
        pages=page_rows,
    )


def collect_docling_memory_profile(path: str | Path) -> list[dict[str, object]]:
    """记录 Docling 关键阶段的 RSS 变化，用于定位内存峰值。"""
    rows: list[dict[str, object]] = []

    def snap(stage: str) -> None:
        rows.append({"stage": stage, "rss_mb": _rss_mb()})

    snap("before_convert")
    _, document = _convert_with_docling(Path(path))
    snap("after_convert")
    doc_dict = _safe_export_dict(document)
    snap("after_export_dict")
    markdown = _safe_export_markdown(document)
    snap("after_export_markdown")
    _pages_from_doc_dict(doc_dict)
    snap("after_pages_from_doc_dict")
    del markdown, doc_dict, document
    gc.collect()
    return rows


def _convert_with_docling(path: Path):
    converter = build_docling_converter()
    result = converter.convert(path)
    document = getattr(result, "document", None)
    if document is None:
        raise RuntimeError(f"Docling conversion returned no document for {path}")
    return result, document


def build_docling_converter():
    """构造稳定的 Docling PDF converter，固定已验证可跑通长财报的配置。"""
    global DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions, PyPdfiumDocumentBackend, _DOC_CONVERTER

    if _DOC_CONVERTER is not None:
        return _DOC_CONVERTER

    if any(symbol is None for symbol in (DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions, PyPdfiumDocumentBackend)):
        try:
            from docling.document_converter import DocumentConverter as _DocumentConverter, PdfFormatOption as _PdfFormatOption
            from docling.datamodel.base_models import InputFormat as _InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions as _PdfPipelineOptions
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend as _PyPdfiumDocumentBackend
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed. Install it in the active Python environment before parsing PDFs."
            ) from exc
        DocumentConverter = _DocumentConverter
        PdfFormatOption = _PdfFormatOption
        InputFormat = _InputFormat
        PdfPipelineOptions = _PdfPipelineOptions
        PyPdfiumDocumentBackend = _PyPdfiumDocumentBackend

    opts = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        force_backend_text=True,
        images_scale=0.5,
    )
    _DOC_CONVERTER = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=opts,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    return _DOC_CONVERTER


def _safe_export_markdown(document: Any) -> str:
    export = getattr(document, "export_to_markdown", None)
    if callable(export):
        try:
            return export()
        except TypeError:
            return export
    return ""


def _safe_export_dict(document: Any) -> dict[str, Any]:
    export = getattr(document, "export_to_dict", None)
    if callable(export):
        payload = export()
        if isinstance(payload, dict):
            return payload
    return {}


def _pages_from_doc_dict(doc_dict: dict[str, Any]) -> list[ParsedPage]:
    pages_node = doc_dict.get("pages")
    if isinstance(pages_node, dict):
        raw_pages = list(pages_node.values())
    elif isinstance(pages_node, list):
        raw_pages = pages_node
    else:
        return []

    raw_pages = [page for page in raw_pages if isinstance(page, dict)]
    if not raw_pages:
        return []

    texts_by_page = _group_items_by_page(doc_dict.get("texts"), kind="text")
    tables_by_page = _group_items_by_page(doc_dict.get("tables"), kind="table")
    figures_by_page = _group_items_by_page(doc_dict.get("pictures"), kind="figure")

    pages: list[ParsedPage] = []
    for idx, page in enumerate(sorted(raw_pages, key=_page_sort_key), start=1):
        page_no_raw = page.get("page_no") or page.get("page") or idx
        page_no = int(page_no_raw) if isinstance(page_no_raw, int) or str(page_no_raw).isdigit() else idx
        text_parts = texts_by_page.get(page_no, [])
        tables = tables_by_page.get(page_no, [])
        figures = figures_by_page.get(page_no, [])
        pages.append(
            ParsedPage(
                page=page_no,
                text=normalize_text("\n".join(text_parts)),
                tables=tables,
                figures=figures,
                parser_name="docling",
                ocr_used=bool(page.get("ocr_used", False)),
                metadata={
                    "table_count": len(tables),
                    "figure_count": len(figures),
                    "table_captions": [item.get("caption", "") for item in tables if item.get("caption")],
                    "figure_captions": [item.get("caption", "") for item in figures if item.get("caption")],
                },
            )
        )
    return pages


def _extract_tables(node: Any) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for item in _walk(node):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("type") or "").lower()
        if label != "table" and "table" not in label:
            continue
        text = _collect_text(item)
        tables.append(
            {
                "label": label or "table",
                "caption": _first_non_empty(item.get("caption"), _join_caption_text(item.get("captions")), item.get("name"), item.get("title")),
                "text": text,
            }
        )
    return _dedupe_rows(tables)


def _extract_figures(node: Any) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    for item in _walk(node):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("type") or "").lower()
        if label not in {"picture", "figure", "chart", "diagram", "image"} and not any(
            key in label for key in ("chart", "figure", "picture", "image", "diagram")
        ):
            continue
        figures.append(
            {
                "label": label or "figure",
                "caption": _first_non_empty(item.get("caption"), _join_caption_text(item.get("captions")), item.get("name"), item.get("title")),
                "text": _collect_text(item),
            }
        )
    return _dedupe_rows(figures)


def _collect_text(node: Any) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    keys = ("text", "content", "orig", "caption", "name", "title", "value")
    for item in _walk(node):
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if isinstance(value, str):
                value = normalize_text(value)
                if value and value not in seen:
                    seen.add(value)
                    parts.append(value)
    return normalize_text("\n".join(parts))


def _group_items_by_page(items: Any, kind: str) -> dict[int, list[Any]]:
    buckets: dict[int, list[Any]] = {}
    if not isinstance(items, list):
        return buckets
    for item in items:
        if not isinstance(item, dict):
            continue
        for page_no in _page_numbers(item):
            buckets.setdefault(page_no, [])
            if kind == "text":
                text = normalize_text(str(item.get("text") or item.get("orig") or ""))
                if text:
                    buckets[page_no].append(text)
            elif kind == "table":
                buckets[page_no].append(
                    {
                        "label": str(item.get("label") or "table"),
                        "caption": _join_caption_text(item.get("captions")),
                        "text": _table_text(item),
                    }
                )
            elif kind == "figure":
                buckets[page_no].append(
                    {
                        "label": str(item.get("label") or "figure"),
                        "caption": _join_caption_text(item.get("captions")),
                        "text": _collect_text(item),
                    }
                )
    return buckets


def _page_numbers(item: dict[str, Any]) -> list[int]:
    prov = item.get("prov")
    page_numbers: list[int] = []
    if isinstance(prov, list):
        for row in prov:
            if not isinstance(row, dict):
                continue
            page_no = row.get("page_no")
            if isinstance(page_no, int):
                page_numbers.append(page_no)
            elif page_no is not None and str(page_no).isdigit():
                page_numbers.append(int(page_no))
    return sorted(set(page_numbers))


def _join_caption_text(captions: Any) -> str:
    parts: list[str] = []
    if isinstance(captions, list):
        for item in captions:
            if isinstance(item, dict):
                value = normalize_text(str(item.get("text") or item.get("orig") or ""))
                if value:
                    parts.append(value)
    return normalize_text("\n".join(parts))


def _table_text(item: dict[str, Any]) -> str:
    data = item.get("data")
    if isinstance(data, dict):
        table_rows = data.get("table_cells") or data.get("grid") or data.get("rows")
        if isinstance(table_rows, list):
            row_texts: list[str] = []
            if table_rows and all(isinstance(cell, dict) for cell in table_rows):
                by_row: dict[int, list[tuple[int, str]]] = {}
                for cell in table_rows:
                    row_idx = int(cell.get("start_row_offset_idx", 0) or 0)
                    col_idx = int(cell.get("start_col_offset_idx", 0) or 0)
                    text = normalize_text(str(cell.get("text") or cell.get("orig") or ""))
                    if text:
                        by_row.setdefault(row_idx, []).append((col_idx, text))
                for row_idx in sorted(by_row):
                    ordered = [text for _, text in sorted(by_row[row_idx], key=lambda x: x[0])]
                    if ordered:
                        row_texts.append(" | ".join(ordered))
            elif table_rows and all(isinstance(row, list) for row in table_rows):
                for row in table_rows:
                    ordered = [normalize_text(str(cell or "")) for cell in row if str(cell or "").strip()]
                    if ordered:
                        row_texts.append(" | ".join(ordered))
            if row_texts:
                return normalize_text("\n".join(row_texts))
    return _collect_text(item)


def _page_sort_key(page: dict[str, Any]) -> int:
    page_no = page.get("page_no") or page.get("page") or 0
    if isinstance(page_no, int):
        return page_no
    return int(page_no) if str(page_no).isdigit() else 0


def _walk(node: Any):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _rss_mb() -> float:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    except Exception:
        return -1.0


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            value = normalize_text(value)
            if value:
                return value
    return ""
