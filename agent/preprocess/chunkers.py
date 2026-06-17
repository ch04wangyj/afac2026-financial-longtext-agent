"""文档分块逻辑。

保险/法规优先按条款切分，合同/财报/研报先按段落切分，再强制控制 chunk 大小。
"""

from __future__ import annotations

import hashlib
import re

from agent.preprocess.domain_cleaning import clean_domain_text
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.domain_rules import infer_candidate_rules
from agent.preprocess.extractors import PageText
from agent.preprocess.normalization import normalize_text
from agent.schemas import Chunk, Document


CLAUSE_RE = re.compile(r"(第[一二三四五六七八九十百千万0-9]+[章节条]|^\d+(?:\.\d+)+)", re.MULTILINE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|％|元|万元|亿元|万|亿)?")
DATE_RE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/]\d{1,2}[-/]\d{1,2}")
HEADING_RE = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节]|[一二三四五六七八九十]+、|\d+(?:\.\d+)*\s+).{0,80}$")


def chunk_document(document: Document, pages: list[PageText], max_chars: int = 1800) -> list[Chunk]:
    """把解析后的页面切成可检索 chunk，并补充数字、日期、条款等元数据。"""
    chunks: list[Chunk] = []
    for page in pages:
        cleaned_text, cleaning_rules = _apply_domain_cleaning(document.domain, page.text)
        parts = _split_page(cleaned_text, document.domain, max_chars=max_chars)
        for idx, part in enumerate(parts):
            part = normalize_text(part)
            if len(part) < 8:
                continue
            chunk = build_text_chunk(
                document=document,
                page=page,
                text=part,
                idx=idx,
                cleaning_rules=cleaning_rules,
                tables=_page_table_texts(page.tables) if idx == 0 else [],
            )
            chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
            chunks.append(chunk)

        for table_idx, table in enumerate(_iter_table_rows(page.tables)):
            chunk = build_table_chunk(
                document=document,
                page=page,
                table=table,
                idx=table_idx,
                cleaning_rules=cleaning_rules,
            )
            if chunk is None:
                continue
            chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
            chunks.append(chunk)

        for figure_idx, figure in enumerate(_iter_figure_rows(page.figures)):
            chunk = build_figure_chunk(
                document=document,
                page=page,
                figure=figure,
                idx=figure_idx,
                cleaning_rules=cleaning_rules,
            )
            if chunk is None:
                continue
            chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
            chunks.append(chunk)
    return chunks


def build_text_chunk(
    document: Document,
    page: PageText,
    text: str,
    idx: int,
    cleaning_rules: list[str],
    tables: list[str],
) -> Chunk:
    section = _infer_section(text)
    clause_id = _infer_clause(text)
    chunk_id = _chunk_id(document.doc_id, page.page, idx, text)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        domain=document.domain,
        page=page.page,
        section=section,
        clause_id=clause_id,
        text=text,
        tables=tables,
        numbers=extract_numbers(text),
        dates=extract_dates(text),
        metadata={
            "title": document.title,
            "path": document.path,
            "cleaning_rules": cleaning_rules,
            "chunk_type": "text",
            "parser_name": page.parser_name,
        },
    )


def build_table_chunk(
    document: Document,
    page: PageText,
    table: dict,
    idx: int,
    cleaning_rules: list[str],
) -> Chunk | None:
    table_text = normalize_text(str(table.get("text") or ""))
    caption = normalize_text(str(table.get("caption") or ""))
    body = normalize_text("\n".join(part for part in (caption, table_text) if part))
    if len(body) < 4:
        return None
    return Chunk(
        chunk_id=_chunk_id(document.doc_id, page.page, 1000 + idx, body),
        doc_id=document.doc_id,
        domain=document.domain,
        page=page.page,
        section="table",
        clause_id="",
        text=body,
        tables=[table_text] if table_text else [],
        numbers=extract_numbers(body),
        dates=extract_dates(body),
        metadata={
            "title": document.title,
            "path": document.path,
            "cleaning_rules": cleaning_rules,
            "chunk_type": "table",
            "caption": caption,
            "parser_name": page.parser_name,
        },
    )


def build_figure_chunk(
    document: Document,
    page: PageText,
    figure: dict,
    idx: int,
    cleaning_rules: list[str],
) -> Chunk | None:
    figure_text = normalize_text(str(figure.get("text") or ""))
    caption = normalize_text(str(figure.get("caption") or ""))
    body = normalize_text("\n".join(part for part in (caption, figure_text) if part))
    if len(body) < 4:
        return None
    return Chunk(
        chunk_id=_chunk_id(document.doc_id, page.page, 2000 + idx, body),
        doc_id=document.doc_id,
        domain=document.domain,
        page=page.page,
        section="figure",
        clause_id="",
        text=body,
        tables=[],
        numbers=extract_numbers(body),
        dates=extract_dates(body),
        metadata={
            "title": document.title,
            "path": document.path,
            "cleaning_rules": cleaning_rules,
            "chunk_type": "figure",
            "caption": caption,
            "parser_name": page.parser_name,
        },
    )


def _apply_domain_cleaning(domain: str, text: str) -> tuple[str, list[str]]:
    """基于样本文本规则推断结果，对页面正文做按域清洗。"""
    cleaning_rules = infer_candidate_rules(domain=domain, sample_text=text).get("cleaning_rules", [])
    return clean_domain_text(domain=domain, text=text, rules=cleaning_rules), cleaning_rules


def extract_numbers(text: str) -> list[str]:
    """抽取金额、比例和普通数字，供检索 boost 与证据压缩使用。"""
    return [m.group(0).strip() for m in NUMBER_RE.finditer(text) if m.group(0).strip()]


def extract_dates(text: str) -> list[str]:
    """抽取常见中文/数字日期。"""
    return [m.group(0).strip() for m in DATE_RE.finditer(text)]


def _split_page(text: str, domain: str, max_chars: int) -> list[str]:
    """根据领域选择条款切分或段落切分。"""
    if not text:
        return []
    if domain in {"insurance", "regulatory"}:
        parts = _split_by_clause(text)
    else:
        parts = _split_by_paragraph(text)
    return _enforce_size(parts, max_chars=max_chars)


def _split_by_clause(text: str) -> list[str]:
    """按“第X条/章”或 1.2.3 这类编号切分。"""
    matches = list(CLAUSE_RE.finditer(text))
    if len(matches) < 2:
        return _split_by_paragraph(text)
    parts: list[str] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        parts.append(text[start:end])
    prefix = text[: matches[0].start()].strip()
    if prefix:
        parts.insert(0, prefix)
    return parts


def _split_by_paragraph(text: str) -> list[str]:
    """按空行切段；没有明显段落时退化到句子级切分。"""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in re.split(r"(?<=[。！？；;])\s*", text) if p.strip()]
    return paras


def _enforce_size(parts: list[str], max_chars: int) -> list[str]:
    """合并短段、切开超长段，避免单块过大影响检索和 Prompt。"""
    output: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > max_chars:
            if buffer:
                output.append(buffer)
                buffer = ""
            output.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
        elif len(buffer) + len(part) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{part}".strip()
        else:
            if buffer:
                output.append(buffer)
            buffer = part
    if buffer:
        output.append(buffer)
    return output


def _infer_clause(text: str) -> str:
    """从 chunk 开头推断条款号。"""
    match = CLAUSE_RE.search(text)
    return match.group(1).strip() if match else ""


def _infer_section(text: str) -> str:
    """从前几行推断章节标题。"""
    for line in text.splitlines()[:5]:
        if HEADING_RE.match(line.strip()):
            return line.strip()
    return ""


def _chunk_id(doc_id: str, page: int | None, idx: int, text: str) -> str:
    """生成稳定 chunk_id，避免原文过长直接进入文件名或索引键。"""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    page_label = "na" if page is None else str(page)
    safe_doc = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8]
    return f"{safe_doc}:p{page_label}:{idx}:{digest}"


def _page_table_texts(tables: list[dict | str]) -> list[str]:
    values: list[str] = []
    for row in tables:
        if isinstance(row, dict):
            text = normalize_text(str(row.get("text") or ""))
        else:
            text = normalize_text(str(row))
        if text:
            values.append(text)
    return values


def _iter_table_rows(tables: list[dict | str]) -> list[dict]:
    rows: list[dict] = []
    for row in tables:
        if isinstance(row, dict):
            rows.append(row)
            continue
        text = normalize_text(str(row))
        if text:
            rows.append({"text": text, "caption": ""})
    return rows


def _iter_figure_rows(figures: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in figures:
        if isinstance(row, dict):
            rows.append(row)
            continue
        text = normalize_text(str(row))
        if text:
            rows.append({"text": text, "caption": ""})
    return rows
