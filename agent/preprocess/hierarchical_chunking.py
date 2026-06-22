"""V13 层级分块：用原子子块检索，用父块恢复必要上下文。

该模块基于现有解析产物重建检索语料，不重新调用 PDF 解析器。这样可以稳定
复用页码、标题和表格，同时修复旧版按 1800 字合并条款造成的证据稀释。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.normalization import normalize_text
from agent.schemas import Chunk


# 数字编号必须以空白或中文标题分隔符结束，避免把 82.91% 等小数识别为条款。
STRICT_CLAUSE_RE = re.compile(r"^(?P<id>第[一二三四五六七八九十百千万0-9]+[章节条款])")
NUMERIC_OUTLINE_RE = re.compile(r"^(?P<id>\d{1,2}(?:\.\d{1,2}){1,4})(?P<rest>\s+.*)$")
HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节]|"
    r"[一二三四五六七八九十百]+、|[（(][一二三四五六七八九十0-9]+[)）]|"
    r"\d{1,2}[、.)）])"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;])")
PAGE_NUMBER_RE = re.compile(r"^(?:第\s*)?\d{1,4}(?:\s*页)?$")
TOC_LEADER_RE = re.compile(r"(?:\.{3,}|…{2,})\s*\d{1,4}\s*$")
NUMERIC_CELL_RE = re.compile(r"[-+()（）]?\d[\d,，]*(?:\.\d+)?(?:%|％|元|万元|亿元|万|亿)?")
SHORT_EVIDENCE_TERMS = (
    "应当",
    "不得",
    "有权",
    "审议",
    "批准",
    "责任",
    "罚款",
    "期限",
    "保险金",
    "营业收入",
    "净利润",
    "现金流",
    "研发",
    "分红",
    "股息",
)


@dataclass(frozen=True)
class HierarchicalChunkConfig:
    """控制原子证据块大小；字符预算比 tokenizer 无关且便于复现。"""

    target_chars: int = 360
    max_chars: int = 520
    min_chars: int = 48


@dataclass(frozen=True)
class _AtomicUnit:
    text: str
    section: str
    clause_id: str
    boundary: bool = False


def build_hierarchical_corpus(
    rows: list[dict],
    config: HierarchicalChunkConfig | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """把旧 chunks 转成父块仓库和仅用于检索的原子子块。"""
    config = config or HierarchicalChunkConfig()
    parents: list[Chunk] = []
    children: list[Chunk] = []
    seen_children: set[tuple[str, str]] = set()

    for row in rows:
        source = Chunk.from_dict(row)
        source_type = str(source.metadata.get("chunk_type", "text"))
        if source_type == "financial_metric_row":
            child = _copy_specialized_child(source, "financial_metric_row")
            _append_unique(children, child, seen_children)
            continue
        if source_type == "figure":
            child = _copy_specialized_child(source, "figure")
            _append_unique(children, child, seen_children)
            continue

        parent = _build_parent(source, source_type)
        parents.append(parent)

        if source_type == "table":
            table_values = source.tables or [source.text]
            generated = _build_table_row_children(source, table_values)
        else:
            generated = _build_atomic_text_children(source, config)
            generated.extend(_build_table_row_children(source, source.tables))
        for child in generated:
            _append_unique(children, child, seen_children)

    return parents, children


def split_atomic_text(
    text: str,
    *,
    inherited_section: str = "",
    inherited_clause: str = "",
    config: HierarchicalChunkConfig | None = None,
) -> list[tuple[str, str, str]]:
    """把页面文本切为不跨条款边界的原子文本块，供测试和构建脚本复用。"""
    config = config or HierarchicalChunkConfig()
    units = _logical_units(text, inherited_section, inherited_clause)
    grouped: list[tuple[str, str, str]] = []
    buffer: list[str] = []
    active_section = inherited_section
    active_clause = inherited_clause

    def flush() -> None:
        nonlocal buffer
        body = normalize_text("".join(buffer))
        if body:
            grouped.extend(_hard_split(body, active_section, active_clause, config.max_chars))
        buffer = []

    for unit in units:
        if unit.boundary:
            flush()
            active_section = unit.section or active_section
            active_clause = unit.clause_id or ""
        candidate = normalize_text("".join([*buffer, unit.text]))
        if buffer and len(candidate) > config.max_chars:
            flush()
        buffer.append(unit.text)
        if len(normalize_text("".join(buffer))) >= config.target_chars:
            flush()
    flush()
    return _merge_tiny_neighbors(grouped, config)


def infer_strict_clause(text: str) -> str:
    """只识别行首条款编号；普通小数和百分比返回空字符串。"""
    first_line = next((line.strip() for line in str(text or "").splitlines() if line.strip()), "")
    match = STRICT_CLAUSE_RE.match(first_line)
    if match:
        return match.group("id")
    numeric = NUMERIC_OUTLINE_RE.match(first_line)
    if numeric is None:
        return ""
    segments = [int(value) for value in numeric.group("id").split(".")]
    rest = numeric.group("rest").strip()
    # 目录层级通常由较小整数构成；金额、比率和连续表格数值不能作为条款。
    if any(value < 1 or value > 30 for value in segments):
        return ""
    if re.match(r"^(?:%|％|元|千元|万元|亿元|万|亿|倍|[-+]?\d)", rest):
        return ""
    return numeric.group("id")


def _build_parent(source: Chunk, source_type: str) -> Chunk:
    metadata = dict(source.metadata)
    metadata.update(
        {
            "chunk_type": "parent",
            "hierarchy_level": "parent",
            "original_chunk_type": source_type,
        }
    )
    return Chunk(
        chunk_id=source.chunk_id,
        doc_id=source.doc_id,
        domain=source.domain,
        page=source.page,
        section=source.section,
        clause_id=infer_strict_clause(source.text) or _safe_legacy_clause(source.clause_id),
        text=source.text,
        tables=list(source.tables),
        numbers=list(source.numbers),
        dates=list(source.dates),
        metadata=metadata,
    )


def _build_atomic_text_children(source: Chunk, config: HierarchicalChunkConfig) -> list[Chunk]:
    output: list[Chunk] = []
    for index, (body, section, clause_id) in enumerate(
        split_atomic_text(
            source.text,
            inherited_section=source.section,
            inherited_clause=infer_strict_clause(source.text) or _safe_legacy_clause(source.clause_id),
            config=config,
        )
    ):
        if len(body) < config.min_chars and not _is_high_signal_short(body):
            continue
        output.append(
            _make_child(
                source,
                body=body,
                child_type="atomic_text",
                child_index=index,
                section=section,
                clause_id=clause_id,
                tables=[],
            )
        )
    return output


def _build_table_row_children(source: Chunk, tables: list[str]) -> list[Chunk]:
    output: list[Chunk] = []
    child_index = 0
    for table_index, raw_table in enumerate(tables):
        lines = [normalize_text(line) for line in str(raw_table or "").splitlines() if normalize_text(line)]
        if not lines:
            continue
        caption = normalize_text(str(source.metadata.get("caption", "")))
        header = _infer_table_header(lines)
        data_lines = lines[1:] if header and len(lines) > 1 else lines
        for row_index, line in enumerate(data_lines):
            if len(line) < 3 or (header and line == header):
                continue
            parts = []
            if caption:
                parts.append(f"表名: {caption}")
            if header:
                parts.append(f"表头: {header}")
            parts.append(f"数据行: {line}")
            body = "\n".join(parts)
            child = _make_child(
                source,
                body=body,
                child_type="table_row",
                child_index=child_index,
                section=source.section or "table",
                clause_id=infer_strict_clause(source.text),
                tables=[line],
            )
            child.metadata.update({"table_index": table_index, "table_row_index": row_index, "table_header": header})
            child.metadata["extra_index_fields"] = build_extra_index_fields(child)
            output.append(child)
            child_index += 1
    return output


def _copy_specialized_child(source: Chunk, child_type: str) -> Chunk:
    metadata = dict(source.metadata)
    metadata.update({"chunk_type": child_type, "hierarchy_level": "child"})
    child = Chunk(
        chunk_id=source.chunk_id,
        doc_id=source.doc_id,
        domain=source.domain,
        page=source.page,
        section=source.section,
        clause_id=infer_strict_clause(source.text) or _safe_legacy_clause(source.clause_id),
        text=source.text,
        tables=list(source.tables),
        numbers=list(source.numbers),
        dates=list(source.dates),
        metadata=metadata,
    )
    child.metadata["extra_index_fields"] = build_extra_index_fields(child)
    return child


def _make_child(
    source: Chunk,
    *,
    body: str,
    child_type: str,
    child_index: int,
    section: str,
    clause_id: str,
    tables: list[str],
) -> Chunk:
    digest = hashlib.sha1(f"{source.chunk_id}:{child_type}:{body}".encode("utf-8")).hexdigest()[:10]
    metadata = {
        "title": source.metadata.get("title", ""),
        "path": source.metadata.get("path", ""),
        "parser_name": source.metadata.get("parser_name", ""),
        "chunk_type": child_type,
        "hierarchy_level": "child",
        "parent_chunk_id": source.chunk_id,
        "atomic_index": child_index,
    }
    child = Chunk(
        chunk_id=f"{source.chunk_id}:a{child_index}:{digest}",
        doc_id=source.doc_id,
        domain=source.domain,
        page=source.page,
        section=section,
        clause_id=clause_id,
        text=body,
        tables=tables,
        numbers=extract_numbers(body),
        dates=extract_dates(body),
        metadata=metadata,
    )
    child.metadata["extra_index_fields"] = build_extra_index_fields(child)
    return child


def _logical_units(text: str, inherited_section: str, inherited_clause: str) -> list[_AtomicUnit]:
    units: list[_AtomicUnit] = []
    pending = ""
    section = inherited_section
    clause_id = inherited_clause

    def flush_pending() -> None:
        nonlocal pending
        for sentence in _split_sentences(pending):
            if sentence:
                units.append(_AtomicUnit(sentence, section, clause_id))
        pending = ""

    for raw_line in normalize_text(text).splitlines():
        line = raw_line.strip()
        if not line or _is_noise_line(line):
            continue
        strict_clause = infer_strict_clause(line)
        is_heading = bool(HEADING_RE.match(line)) and len(line) <= 100
        if strict_clause or is_heading:
            flush_pending()
            if is_heading:
                section = line
            clause_id = strict_clause or clause_id
            units.append(_AtomicUnit(line, section, clause_id, boundary=True))
            continue
        pending += line
        if line.endswith(("。", "！", "？", "；", ";")):
            flush_pending()
    flush_pending()
    return units


def _split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part for part in SENTENCE_SPLIT_RE.split(normalized) if part]


def _hard_split(text: str, section: str, clause_id: str, max_chars: int) -> list[tuple[str, str, str]]:
    if len(text) <= max_chars:
        return [(text, section, clause_id)]
    output: list[tuple[str, str, str]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            nearby = max(text.rfind(mark, start, end) for mark in ("。", "；", ";", "，", ","))
            if nearby > start + max_chars // 2:
                end = nearby + 1
        output.append((text[start:end], section, clause_id))
        start = end
    return output


def _merge_tiny_neighbors(
    chunks: list[tuple[str, str, str]],
    config: HierarchicalChunkConfig,
) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    for body, section, clause_id in chunks:
        if (
            output
            and len(body) < config.min_chars
            and output[-1][1:] == (section, clause_id)
            and len(output[-1][0]) + len(body) <= config.max_chars
        ):
            previous, _, _ = output[-1]
            output[-1] = (normalize_text(previous + body), section, clause_id)
        else:
            output.append((body, section, clause_id))
    return output


def _infer_table_header(lines: list[str]) -> str:
    first = lines[0]
    cells = [cell.strip() for cell in first.split("|") if cell.strip()]
    if len(cells) < 2:
        return ""
    numeric_cells = sum(bool(NUMERIC_CELL_RE.fullmatch(cell)) for cell in cells)
    return first if numeric_cells < max(1, len(cells) // 2) else ""


def _is_noise_line(line: str) -> bool:
    compact = "".join(line.split())
    return compact in {"目录", "目次"} or bool(PAGE_NUMBER_RE.fullmatch(compact)) or bool(TOC_LEADER_RE.search(line))


def _safe_legacy_clause(value: str) -> str:
    """旧索引中的纯小数 clause_id 不可信，只继承明确的中文章节条编号。"""
    value = str(value or "").strip()
    return value if value.startswith("第") else ""


def _is_high_signal_short(text: str) -> bool:
    """短法条、列表项和财务事实不能因字符少而被删除。"""
    stripped = text.strip()
    return bool(
        extract_numbers(stripped)
        or infer_strict_clause(stripped)
        or HEADING_RE.match(stripped)
        or any(term in stripped for term in SHORT_EVIDENCE_TERMS)
    )


def _append_unique(children: list[Chunk], child: Chunk, seen: set[tuple[str, str]]) -> None:
    key = (child.doc_id, "".join(child.text.split()))
    if not key[1] or key in seen:
        return
    seen.add(key)
    children.append(child)
