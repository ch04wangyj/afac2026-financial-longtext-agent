"""V14 无模型 PDF 版面与表格解析。

本模块只使用 PDF 自带的文字坐标和矢量线，不调用 OCR、VLM 或其他模型。
解析结果作为 V13 语料的增量补充，避免一次替换旧文本导致召回回归。
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.preprocess.normalization import compact_for_search, normalize_text
from agent.schemas import Chunk, Document


NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?(?:%|％|元|万元|亿元|万|亿|倍|pp)?")
YEAR_RE = re.compile(r"20\d{2}\s*年?")
UNIT_RE = re.compile(
    r"(?:单位|金额单位|币种)\s*(?:[:：]|为)?\s*"
    r"((?:人民币)?(?:百万元|千元|万元|亿元|港元|美元|元))"
)
PAGE_NUMBER_RE = re.compile(r"^[-—–\s]*(?:第\s*)?\d{1,4}(?:\s*页)?[-—–\s]*$")
FINANCIAL_TERMS = (
    "营业收入",
    "净利润",
    "现金流",
    "研发投入",
    "分红",
    "股息",
    "总资产",
    "净资产",
    "同比",
    "本期",
    "上期",
    "2024年",
    "2025年",
)


@dataclass(frozen=True)
class LayoutParseConfig:
    """控制增量块和表格检测规模。"""

    max_block_chars: int = 720
    min_block_chars: int = 24
    margin_ratio: float = 0.09
    recurring_margin_ratio: float = 0.28
    min_table_rows: int = 2
    max_table_rows: int = 180
    detect_borderless_tables: bool = True
    # V15 B1: 跨页表格链式继承
    enable_chain_continuation: bool = True
    max_continuation_gap: int = 2  # 允许中间隔页的最大页数
    # V15 B2: 双栏 X 坐标聚类
    enable_column_clustering: bool = True
    # V15 B3: 页眉页脚模糊匹配
    enable_fuzzy_margin: bool = True
    fuzzy_margin_threshold: float = 0.75  # 字符相似度阈值
    # V15 B4: 多级表头展开
    enable_multilevel_header: bool = True


@dataclass(frozen=True)
class VisualLine:
    """一条带坐标和逻辑单元格的 PDF 文本行。"""

    bbox: tuple[float, float, float, float]
    text: str
    cells: tuple[str, ...]
    max_font_size: float = 0.0
    bold: bool = False


@dataclass
class LayoutTable:
    """统一的有框/无框表格结构。"""

    page: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]]
    source: str
    caption: str = ""
    unit: str = ""
    header: list[str] = field(default_factory=list)
    continuation: bool = False
    table_id: str = ""


@dataclass
class LayoutPage:
    """单页版面审计结果。"""

    page: int
    width: float
    height: float
    blocks: list[tuple[tuple[float, float, float, float], str]]
    lines: list[VisualLine]
    tables: list[LayoutTable]
    two_column: bool
    removed_margin_lines: int = 0


def parse_pdf_layout(
    path: Path,
    *,
    config: LayoutParseConfig | None = None,
) -> list[LayoutPage]:
    """解析 PDF 的文字块、阅读栏和表格，不使用任何推理模型。"""
    import fitz

    config = config or LayoutParseConfig()
    pages: list[LayoutPage] = []
    with fitz.open(path) as document:
        recurring = _recurring_margin_signatures(document, config)
        if config.enable_fuzzy_margin:
            recurring = recurring | _detect_recurring_headers_fuzzy(document, config)
        # 跨页续表跟踪：保留最近几个页的表格，支持隔页继承
        recent_tables: list[LayoutTable] = []
        for page_index, page in enumerate(document, start=1):
            lines, removed = _extract_visual_lines(page, recurring, config)
            blocks = _extract_text_blocks(page, recurring, config)
            two_column = detect_two_column(blocks, float(page.rect.width))
            if config.enable_column_clustering and not two_column:
                two_column = _detect_columns_by_clustering(blocks, float(page.rect.width))
            tables = extract_page_tables(page, lines, page_index, config)
            for table_index, table in enumerate(tables):
                table.table_id = _stable_table_id(path, page_index, table_index, table)
                if config.enable_chain_continuation:
                    _apply_chain_continuation(table, recent_tables, config)
                else:
                    if _is_continuation(recent_tables[-1] if recent_tables else None, table):
                        table.continuation = True
                        if not table.header:
                            table.header = list(recent_tables[-1].header)
                        if not table.caption:
                            table.caption = recent_tables[-1].caption
                        if not table.unit:
                            table.unit = recent_tables[-1].unit
                recent_tables.append(table)
                # 只保留最近 max_continuation_gap+1 页的表格
                max_keep = config.max_continuation_gap + 2
                if len(recent_tables) > max_keep * 3:
                    recent_tables = recent_tables[-max_keep * 3:]
            pages.append(
                LayoutPage(
                    page=page_index,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    blocks=blocks,
                    lines=lines,
                    tables=tables,
                    two_column=two_column,
                    removed_margin_lines=removed,
                )
            )
    return pages


def build_layout_supplement_chunks(
    document: Document,
    *,
    config: LayoutParseConfig | None = None,
) -> list[Chunk]:
    """把版面解析结果转为可直接并入 BM25F 的增量证据块。"""
    config = config or LayoutParseConfig()
    output: list[Chunk] = []
    seen: set[str] = set()
    for page in parse_pdf_layout(document.path_obj, config=config):
        for table in page.tables:
            for row_index, row in enumerate(_data_rows(table)):
                body = format_table_row(table, row)
                chunk = _make_chunk(
                    document,
                    page=page.page,
                    body=body,
                    chunk_type="layout_table_row",
                    index=len(output),
                    section=table.caption or "table",
                    tables=[" | ".join(row)],
                    metadata={
                        "layout_source": table.source,
                        "table_id": table.table_id,
                        "table_header": format_table_header(table.header, table=table),
                        "table_unit": table.unit,
                        "table_continuation": table.continuation,
                        "row_index": row_index,
                        "bbox": list(table.bbox),
                    },
                )
                _append_unique(output, chunk, seen)

        for block_index, (bbox, text) in enumerate(page.blocks):
            if not _keep_layout_block(text, page.two_column, document.domain, config):
                continue
            for split_index, body in enumerate(_split_block(text, config.max_block_chars)):
                chunk = _make_chunk(
                    document,
                    page=page.page,
                    body=body,
                    chunk_type="layout_text",
                    index=100_000 + block_index * 100 + split_index,
                    section=_infer_block_section(page.lines, bbox),
                    tables=[],
                    metadata={
                        "bbox": list(bbox),
                        "two_column_page": page.two_column,
                        "removed_margin_lines": page.removed_margin_lines,
                    },
                )
                _append_unique(output, chunk, seen)
    return output


def extract_page_tables(
    page: Any,
    lines: list[VisualLine],
    page_number: int,
    config: LayoutParseConfig,
) -> list[LayoutTable]:
    """优先提取矢量线表格，再补充坐标对齐的无框表格。"""
    output = _extract_ruled_tables(page, lines, page_number, config)
    if config.detect_borderless_tables:
        borderless = _extract_borderless_tables(lines, page_number, config)
        for candidate in borderless:
            if any(_bbox_overlap(candidate.bbox, existing.bbox) >= 0.65 for existing in output):
                continue
            output.append(candidate)
    output.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    return output


def detect_two_column(
    blocks: list[tuple[tuple[float, float, float, float], str]],
    page_width: float,
) -> bool:
    """用左右窄块的字符覆盖和纵向重叠判断双栏，不依赖版面模型。"""
    if page_width <= 0:
        return False
    left: list[tuple[tuple[float, float, float, float], str]] = []
    right: list[tuple[tuple[float, float, float, float], str]] = []
    center = page_width / 2
    for item in blocks:
        (x0, _, x1, _), text = item
        if len(compact_for_search(text)) < 20 or x1 - x0 > page_width * 0.64:
            continue
        midpoint = (x0 + x1) / 2
        if midpoint < center * 0.92:
            left.append(item)
        elif midpoint > center * 1.08:
            right.append(item)
    if len(left) < 2 or len(right) < 2:
        return False
    left_chars = sum(len(text) for _, text in left)
    right_chars = sum(len(text) for _, text in right)
    left_y = (min(box[1] for box, _ in left), max(box[3] for box, _ in left))
    right_y = (min(box[1] for box, _ in right), max(box[3] for box, _ in right))
    overlap = max(0.0, min(left_y[1], right_y[1]) - max(left_y[0], right_y[0]))
    span = max(1.0, min(left_y[1] - left_y[0], right_y[1] - right_y[0]))
    return left_chars >= 120 and right_chars >= 120 and overlap / span >= 0.35


def format_table_row(table: LayoutTable, row: list[str]) -> str:
    """每个数据行重复标题、单位和表头，保证脱离原页后仍可解释。"""
    parts: list[str] = []
    if table.caption:
        parts.append(f"表名: {table.caption}")
    if table.unit:
        parts.append(f"单位: {table.unit}")
    if table.header:
        parts.append(f"表头: {format_table_header(table.header, table=table)}")
    if table.continuation:
        parts.append("跨页续表: 是")
    parts.append(f"数据行: {' | '.join(row)}")
    return "\n".join(parts)


def format_table_header(header: list[str], *, table: LayoutTable | None = None) -> str:
    """把两层年度表头展开到数据列，避免年份和金额/占比顺序歧义。

    V15 B4: 如果传入 table 且有多行表头，先尝试多级展开。
    """
    # V15 B4: 多级表头展开 — 只在前几行都是非数值行时触发
    if table is not None and len(table.rows) >= 3:
        multi_header_rows = _detect_multilevel_header_rows(table.rows)
        if len(multi_header_rows) >= 2:
            _, data_rows = _split_header_rows(table.rows)
            expanded = _expand_multilevel_header(multi_header_rows, data_rows or table.rows[len(multi_header_rows):])
            if expanded and len(expanded) > 1:
                return " | ".join(expanded)
    years = [value for value in header if YEAR_RE.search(value)]
    changes = [value for value in header if any(term in value for term in ("同比", "增减", "变化", "变动"))]
    subheaders = [value for value in header if value not in years and value not in changes]
    if len(years) == 2 and len(subheaders) == 2:
        expanded = [
            f"{years[0]}-{subheaders[0]}",
            f"{years[0]}-{subheaders[1]}",
            f"{years[1]}-{subheaders[0]}",
            f"{years[1]}-{subheaders[1]}",
            *(changes[:1] or ["同比增减"]),
        ]
        return " | ".join(expanded)
    return " | ".join(header)


def _detect_multilevel_header_rows(rows: list[list[str]]) -> list[list[str]]:
    """V15 B4: 检测前几行是否是多级表头（非数值行）。

    返回被判定为表头的行列表。只有当连续 2+ 行都是非数值行时才返回多行。
    """
    header_rows: list[list[str]] = []
    for index, row in enumerate(rows[:4]):
        if not row:
            continue
        numeric_count = sum(bool(NUMBER_RE.fullmatch(cell.replace(" ", "").replace(",", ""))) for cell in row)
        has_year = any(YEAR_RE.search(cell) for cell in row)
        # 表头行：没有纯数值单元格，或有年份
        if numeric_count == 0 or (has_year and numeric_count <= 1 and index == 0):
            header_rows.append(row)
            continue
        break
    return header_rows if len(header_rows) >= 2 else []


def _recurring_margin_signatures(document: Any, config: LayoutParseConfig) -> set[str]:
    counts: Counter[str] = Counter()
    page_count = len(document)
    for page in document:
        height = float(page.rect.height)
        for block in page.get_text("blocks", sort=True):
            if len(block) < 7 or int(block[6]) != 0:
                continue
            y0, y1 = float(block[1]), float(block[3])
            if y1 > height * config.margin_ratio and y0 < height * (1 - config.margin_ratio):
                continue
            for line in str(block[4] or "").splitlines():
                signature = _margin_signature(line)
                if signature:
                    counts[signature] += 1
    threshold = max(3, int(page_count * config.recurring_margin_ratio))
    return {signature for signature, count in counts.items() if count >= threshold}


def _extract_visual_lines(
    page: Any,
    recurring: set[str],
    config: LayoutParseConfig,
) -> tuple[list[VisualLine], int]:
    lines: list[VisualLine] = []
    removed = 0
    height = float(page.rect.height)
    payload = page.get_text("dict", sort=False)
    for block in payload.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for raw_line in block.get("lines", []):
            spans = [span for span in raw_line.get("spans", []) if normalize_text(str(span.get("text", "")))]
            if not spans:
                continue
            spans.sort(key=lambda span: float(span.get("bbox", [0, 0, 0, 0])[0]))
            cells = _spans_to_cells(spans)
            text = normalize_text("".join(str(span.get("text", "")) for span in spans))
            bbox = tuple(float(value) for value in raw_line.get("bbox", (0, 0, 0, 0)))
            if _is_margin_text(text, bbox, height, recurring, config):
                removed += 1
                continue
            fonts = [float(span.get("size", 0.0) or 0.0) for span in spans]
            flags = [int(span.get("flags", 0) or 0) for span in spans]
            lines.append(
                VisualLine(
                    bbox=bbox,
                    text=text,
                    cells=tuple(cells),
                    max_font_size=max(fonts, default=0.0),
                    bold=any(flag & 16 for flag in flags),
                )
            )
    lines.sort(key=lambda line: ((line.bbox[1] + line.bbox[3]) / 2, line.bbox[0]))
    return _merge_visual_rows(lines), removed


def _extract_text_blocks(
    page: Any,
    recurring: set[str],
    config: LayoutParseConfig,
) -> list[tuple[tuple[float, float, float, float], str]]:
    height = float(page.rect.height)
    output: list[tuple[tuple[float, float, float, float], str]] = []
    for block in page.get_text("blocks", sort=False):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        bbox = tuple(float(value) for value in block[:4])
        kept_lines = [
            normalize_text(line)
            for line in str(block[4] or "").splitlines()
            if normalize_text(line) and not _is_margin_text(line, bbox, height, recurring, config)
        ]
        text = normalize_text("\n".join(kept_lines))
        if text and not PAGE_NUMBER_RE.fullmatch(text):
            output.append((bbox, text))
    return output


def _extract_ruled_tables(
    page: Any,
    lines: list[VisualLine],
    page_number: int,
    config: LayoutParseConfig,
) -> list[LayoutTable]:
    try:
        finder = page.find_tables(strategy="lines")
    except Exception:
        return []
    output: list[LayoutTable] = []
    for raw_table in [] if finder is None else finder.tables:
        raw_rows = raw_table.extract() or []
        rows = _normalize_grid_rows(raw_rows)
        if not _valid_table(rows, config):
            continue
        bbox = tuple(float(value) for value in raw_table.bbox)
        caption, unit = _caption_and_unit(lines, bbox)
        header, _ = _split_header_rows(rows)
        output.append(
            LayoutTable(
                page=page_number,
                bbox=bbox,
                rows=rows[: config.max_table_rows],
                source="pymupdf_lines",
                caption=caption,
                unit=unit,
                header=header,
            )
        )
    return output


def _extract_borderless_tables(
    lines: list[VisualLine],
    page_number: int,
    config: LayoutParseConfig,
) -> list[LayoutTable]:
    """从对齐文本行中恢复无线框财务表；连续行共享年份表头。"""
    output: list[LayoutTable] = []
    active: list[VisualLine] = []
    active_header: list[str] = []
    active_caption = ""
    active_unit = ""

    def flush() -> None:
        nonlocal active, active_header, active_caption, active_unit
        if len(active) >= config.min_table_rows:
            rows = [list(line.cells) for line in active if len(line.cells) >= 2]
            if _valid_table(rows, config):
                bbox = _union_bbox(line.bbox for line in active)
                output.append(
                    LayoutTable(
                        page=page_number,
                        bbox=bbox,
                        rows=[active_header, *rows] if active_header else rows,
                        source="coordinate_alignment",
                        caption=active_caption,
                        unit=active_unit,
                        header=list(active_header),
                    )
                )
        active = []
        active_header = []
        active_caption = ""
        active_unit = ""

    for index, line in enumerate(lines):
        cells = list(line.cells)
        year_count = len(YEAR_RE.findall(line.text))
        numeric_cells = sum(bool(NUMBER_RE.search(cell)) for cell in cells)
        if year_count >= 2 and len(cells) >= 2:
            flush()
            active_header = cells
            active_caption, active_unit = _nearby_caption_unit(lines, index)
            continue
        is_data_row = len(cells) >= 3 and numeric_cells >= 2
        if active_header and is_data_row:
            active.append(line)
            continue
        if active and is_data_row and line.bbox[1] - active[-1].bbox[3] < 24:
            active.append(line)
            continue
        if active:
            flush()
    flush()
    return output


def _spans_to_cells(spans: list[dict[str, Any]]) -> list[str]:
    cells: list[str] = []
    current = ""
    previous_x1: float | None = None
    previous_size = 10.0
    for span in spans:
        text = normalize_text(str(span.get("text", "")))
        if not text:
            continue
        x0, _, x1, _ = (float(value) for value in span.get("bbox", (0, 0, 0, 0)))
        size = float(span.get("size", 10.0) or 10.0)
        gap = 0.0 if previous_x1 is None else x0 - previous_x1
        threshold = max(6.0, min(previous_size, size) * 1.15)
        if current and gap > threshold:
            cells.append(compact_for_search(current))
            current = text
        else:
            current = f"{current}{text}"
        previous_x1 = x1
        previous_size = size
    if current:
        cells.append(compact_for_search(current))
    return [cell for cell in cells if cell]


def _merge_visual_rows(lines: list[VisualLine]) -> list[VisualLine]:
    """把 PDF 内部拆开的同一视觉行按 Y 轴和非重叠 X 区间重新聚合。"""
    groups: list[list[VisualLine]] = []
    for line in lines:
        center = (line.bbox[1] + line.bbox[3]) / 2
        target: list[VisualLine] | None = None
        for group in reversed(groups[-3:]):
            group_center = sum((item.bbox[1] + item.bbox[3]) / 2 for item in group) / len(group)
            if abs(center - group_center) > 2.4:
                continue
            if any(_horizontal_overlap(line.bbox, item.bbox) > 0.35 for item in group):
                continue
            target = group
            break
        if target is None:
            groups.append([line])
        else:
            target.append(line)

    output: list[VisualLine] = []
    for group in groups:
        ordered = sorted(group, key=lambda item: item.bbox[0])
        cells = [cell for item in ordered for cell in item.cells]
        output.append(
            VisualLine(
                bbox=_union_bbox(item.bbox for item in ordered),
                text=normalize_text(" ".join(item.text for item in ordered)),
                cells=tuple(cells),
                max_font_size=max(item.max_font_size for item in ordered),
                bold=any(item.bold for item in ordered),
            )
        )
    output.sort(key=lambda line: ((line.bbox[1] + line.bbox[3]) / 2, line.bbox[0]))
    return output


def _dense_row(row: Iterable[Any]) -> list[str]:
    output: list[str] = []
    for cell in row:
        value = _clean_cell(str(cell or ""))
        if value:
            output.append(value)
    return output


def _normalize_grid_rows(raw_rows: list[list[Any]]) -> list[list[str]]:
    """合并由纵向合并单元格造成的互补稀疏行，再删除空列占位。"""
    sparse_rows = [[_clean_cell(str(cell or "")) for cell in row] for row in raw_rows]
    merged: list[list[str] | None] = []
    for row in sparse_rows:
        occupied = {index for index, value in enumerate(row) if value}
        if not occupied:
            merged.append(None)
            continue
        previous = merged[-1] if merged else None
        previous_occupied = {index for index, value in enumerate(previous or []) if value}
        previous_has_number = any(NUMBER_RE.search(value) for value in (previous or []) if value)
        if (
            previous is not None
            and len(previous) == len(row)
            and previous_occupied.isdisjoint(occupied)
            and previous_has_number
        ):
            merged[-1] = [left or right for left, right in zip(previous, row)]
        else:
            merged.append(row)
    return [_dense_row(row) for row in merged if row is not None and _dense_row(row)]


def _valid_table(rows: list[list[str]], config: LayoutParseConfig) -> bool:
    if not (config.min_table_rows <= len(rows) <= config.max_table_rows):
        return False
    nonempty = sum(len(row) for row in rows)
    width = max((len(row) for row in rows), default=0)
    numeric = sum(bool(NUMBER_RE.search(cell)) for row in rows for cell in row)
    has_financial_term = any(term in cell for term in FINANCIAL_TERMS for row in rows for cell in row)
    return width >= 2 and nonempty >= 4 and (numeric >= 2 or has_financial_term)


def _split_header_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    start = 0
    for index, row in enumerate(rows[:4]):
        numeric = sum(bool(NUMBER_RE.fullmatch(cell.replace(" ", ""))) for cell in row)
        has_year = any(YEAR_RE.search(cell) for cell in row)
        if index == 0 and len(row) == 1 and not has_year:
            headers.extend(row)
            start = index + 1
            continue
        if has_year or numeric == 0:
            headers.extend(row)
            start = index + 1
            continue
        break
    return list(dict.fromkeys(headers)), rows[start:]


def _data_rows(table: LayoutTable) -> list[list[str]]:
    _, rows = _split_header_rows(table.rows)
    rows = rows or table.rows
    return [
        row
        for row in rows
        if len(row) >= 2
        or any(NUMBER_RE.search(cell) or any(term in cell for term in FINANCIAL_TERMS) for cell in row)
    ]


def _caption_and_unit(
    lines: list[VisualLine],
    bbox: tuple[float, float, float, float],
) -> tuple[str, str]:
    candidates = [line for line in lines if line.bbox[3] <= bbox[1] + 2 and bbox[1] - line.bbox[3] <= 110]
    caption = _choose_caption(candidates)
    unit = ""
    for line in reversed(candidates[-8:]):
        unit_match = UNIT_RE.search(line.text)
        if unit_match and not unit:
            unit = compact_for_search(unit_match.group(1))
    return caption, unit


def _nearby_caption_unit(lines: list[VisualLine], index: int) -> tuple[str, str]:
    candidates = lines[max(0, index - 12) : index]
    caption = _choose_caption(candidates)
    unit = ""
    for line in reversed(candidates):
        match = UNIT_RE.search(line.text)
        if match and not unit:
            unit = compact_for_search(match.group(1))
    return caption, unit


def _looks_like_caption(line: VisualLine) -> bool:
    text = compact_for_search(line.text)
    return bool(
        2 <= len(text) <= 90
        and not PAGE_NUMBER_RE.fullmatch(text)
        and (
            line.bold
            or line.max_font_size >= 11.0
            or text.startswith(("表", "附表", "注:"))
            or (len(text) <= 50 and any(term in text for term in FINANCIAL_TERMS))
        )
    )


def _choose_caption(lines: list[VisualLine]) -> str:
    scored: list[tuple[float, int, str]] = []
    for index, line in enumerate(lines):
        if not _looks_like_caption(line):
            continue
        text = compact_for_search(line.text)
        score = 0.0
        if line.max_font_size >= 13:
            score += 4.0
        elif line.max_font_size >= 11:
            score += 1.5
        if line.bold:
            score += 3.0
        if text.startswith(("表", "附表")):
            score += 2.0
        if len(text) <= 50 and any(term in text for term in FINANCIAL_TERMS):
            score += 1.0
        scored.append((score, index, text))
    return max(scored, default=(0.0, 0, ""))[2]


def _is_continuation(previous: LayoutTable | None, current: LayoutTable) -> bool:
    if previous is None or current.page != previous.page + 1:
        return False
    previous_width = _median_row_width(previous.rows)
    current_width = _median_row_width(current.rows)
    if previous_width < 2 or previous_width != current_width:
        return False
    first = current.rows[0] if current.rows else []
    first_has_header = any(YEAR_RE.search(cell) for cell in first)
    shared_header = set(previous.header) & set(current.header)
    same_header = bool(
        shared_header
        and len(shared_header) >= max(1, min(len(previous.header), len(current.header)) // 2)
    )
    explicit_continuation = "续" in current.caption or "续" in "".join(current.header)
    geometric_continuation = previous.bbox[3] >= 620 and current.bbox[1] <= 130
    first_has_value = sum(bool(NUMBER_RE.search(cell)) for cell in first) >= 1
    return explicit_continuation or same_header or (
        geometric_continuation and not first_has_header and first_has_value
    )


def _apply_chain_continuation(
    current: LayoutTable,
    recent_tables: list[LayoutTable],
    config: LayoutParseConfig,
) -> None:
    """V15 B1: 链式跨页续表检测，支持隔页继承和表头变异。

    从最近的表格往回找，允许跳过最多 max_continuation_gap 页。
    """
    if not recent_tables:
        return
    # 只看当前页之前 max_continuation_gap+1 页内的表格
    min_page = current.page - config.max_continuation_gap - 1
    candidates = [t for t in recent_tables if t.page >= min_page and t.page < current.page]
    if not candidates:
        return
    # 按页码降序，优先匹配最近的
    candidates.sort(key=lambda t: t.page, reverse=True)
    for candidate in candidates:
        if _is_continuation_relaxed(candidate, current, config):
            current.continuation = True
            if not current.header:
                current.header = list(candidate.header)
            if not current.caption:
                current.caption = candidate.caption
            if not current.unit:
                current.unit = candidate.unit
            return


def _is_continuation_relaxed(
    previous: LayoutTable,
    current: LayoutTable,
    config: LayoutParseConfig,
) -> bool:
    """V15 B1: 放宽的续表判断，容忍表头变异和行宽差异。"""
    page_gap = current.page - previous.page
    if page_gap <= 0 or page_gap > config.max_continuation_gap + 1:
        return False
    # 显式续表标记
    explicit_continuation = "续" in current.caption or "续" in "".join(current.header)
    if explicit_continuation:
        return True
    # 表头相似度（允许列顺序变化、单位行变化）
    if previous.header and current.header:
        shared = set(previous.header) & set(current.header)
        min_len = min(len(previous.header), len(current.header))
        if min_len > 0 and len(shared) >= max(1, min_len // 2):
            return True
    # 行宽容忍：允许差 1 列（新增/删除列）
    prev_width = _median_row_width(previous.rows)
    curr_width = _median_row_width(current.rows)
    if prev_width >= 2 and abs(prev_width - curr_width) <= 1:
        # 几何条件：前表底部接近页底 + 当前表顶部接近页顶
        geometric = previous.bbox[3] >= 620 and current.bbox[1] <= 130
        first = current.rows[0] if current.rows else []
        first_has_value = sum(bool(NUMBER_RE.search(cell)) for cell in first) >= 1
        first_has_header = any(YEAR_RE.search(cell) for cell in first)
        if geometric and first_has_value and not first_has_header:
            return True
    return False


def _detect_columns_by_clustering(
    blocks: list[tuple[tuple[float, float, float, float], str]],
    page_width: float,
) -> bool:
    """V15 B2: 基于文本块 X 坐标聚类检测双栏。

    不要求两侧垂直重叠，处理不对称双栏和混合版面。
    """
    if page_width <= 0 or len(blocks) < 4:
        return False
    center = page_width / 2
    # 收集所有窄块的 X 中点
    midpoints: list[float] = []
    widths: list[float] = []
    for (x0, _, x1, _), text in blocks:
        if len(compact_for_search(text)) < 15:
            continue
        block_width = x1 - x0
        if block_width > page_width * 0.64:
            continue  # 跨栏块跳过
        midpoints.append((x0 + x1) / 2)
        widths.append(block_width)
    if len(midpoints) < 4:
        return False
    # 按中点分到左右两组
    left = [m for m in midpoints if m < center * 0.92]
    right = [m for m in midpoints if m > center * 1.08]
    if len(left) < 2 or len(right) < 2:
        return False
    # 不对称双栏：只要两侧各有足够多的块即可
    # 但要排除"单栏左对齐+右侧少量注释"的情况
    total = len(left) + len(right)
    min_ratio = 0.2  # 较少一侧至少占 20%
    if len(left) / total < min_ratio or len(right) / total < min_ratio:
        return False
    # 验证 X 中心分离度
    left_center = sum(left) / len(left)
    right_center = sum(right) / len(right)
    if right_center - left_center < page_width * 0.2:
        return False
    return True


def _detect_recurring_headers_fuzzy(document: Any, config: LayoutParseConfig) -> set[str]:
    """V15 B3: 模糊匹配跨页页眉页脚。

    处理页码变体、日期变体，以及不完全相同但高度相似的重复行。
    """
    from difflib import SequenceMatcher

    page_count = len(document)
    if page_count < 4:
        return set()
    # 收集每页页眉页脚候选行
    margin_lines: list[list[str]] = []
    for page in document:
        height = float(page.rect.height)
        page_lines: list[str] = []
        for block in page.get_text("blocks", sort=True):
            if len(block) < 7 or int(block[6]) != 0:
                continue
            y0, y1 = float(block[1]), float(block[3])
            if y1 > height * config.margin_ratio and y0 < height * (1 - config.margin_ratio):
                continue
            for line in str(block[4] or "").splitlines():
                text = normalize_text(line)
                if text and len(text) >= 2:
                    page_lines.append(_fuzzy_margin_signature(text))
        margin_lines.append(page_lines)
    # 跨页模糊匹配
    threshold = max(3, int(page_count * config.recurring_margin_ratio))
    result: set[str] = set()
    # 统计每个签名出现的页数
    signature_pages: dict[str, set[int]] = {}
    for page_idx, lines in enumerate(margin_lines):
        for sig in lines:
            if not sig:
                continue
            signature_pages.setdefault(sig, set()).add(page_idx)
    # 精确匹配
    for sig, pages in signature_pages.items():
        if len(pages) >= threshold:
            result.add(sig)
    # 模糊匹配：对未达到阈值的签名，找相似签名合并
    all_sigs = list(signature_pages.keys())
    for i, sig_a in enumerate(all_sigs):
        if sig_a in result:
            continue
        merged_pages = set(signature_pages[sig_a])
        for j, sig_b in enumerate(all_sigs):
            if i == j or sig_b in result:
                continue
            similarity = SequenceMatcher(None, sig_a, sig_b).ratio()
            if similarity >= config.fuzzy_margin_threshold:
                merged_pages |= signature_pages[sig_b]
        if len(merged_pages) >= threshold:
            result.add(sig_a)
    return result


def _fuzzy_margin_signature(text: str) -> str:
    """V15 B3: 生成模糊页眉页脚签名，归一化页码和日期变体。"""
    compact = re.sub(r"\s+", "", normalize_text(text)).casefold()
    # 归一化页码变体: "第X页"、"Page X of Y"、纯数字
    compact = re.sub(r"第\s*\d+\s*页", "第#页", compact)
    compact = re.sub(r"page\s*\d+", "page#", compact)
    compact = re.sub(r"^\d{1,4}$", "#", compact)
    # 归一化日期变体: "2026年X月X日"、"2026-X-X"
    compact = re.sub(r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{0,2}\s*日?", "日期", compact)
    compact = re.sub(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", "日期", compact)
    compact = re.sub(r"\d+", "#", compact)
    return compact if len(compact) >= 2 else ""


def _expand_multilevel_header(
    header_rows: list[list[str]],
    data_rows: list[list[str]],
) -> list[str]:
    """V15 B4: 展开多级表头到数据列。

    支持三级表头（大类→子类→年份）、合并单元格（rowspan/colspan）和嵌套结构。
    返回展开后的单行表头列表，长度与 data_rows 的列数对齐。
    """
    if not header_rows or not data_rows:
        return header_rows[-1] if header_rows else []
    # 数据行列数作为目标宽度
    target_width = max(len(row) for row in data_rows) if data_rows else 0
    if target_width == 0:
        return header_rows[-1] if header_rows else []
    # 单行表头直接返回
    if len(header_rows) == 1:
        header = header_rows[0]
        if len(header) == target_width:
            return header
        # 补齐或截断
        return (header + [""] * target_width)[:target_width]
    # 多行表头：从上到下逐层展开
    # 每一行的空单元格先继承同行前一个非空值（水平 colspan），再继承上方同列值（垂直 rowspan）
    expanded: list[list[str]] = []
    for row_idx, row in enumerate(header_rows):
        expanded_row: list[str] = []
        last_non_empty = ""
        for col_idx in range(max(len(row), target_width)):
            cell = row[col_idx] if col_idx < len(row) else ""
            if not cell:
                # 水平 colspan：继承同行前一个非空值
                cell = last_non_empty
            else:
                last_non_empty = cell
            # 垂直 rowspan：如果仍为空且上方同列有值，继承上方
            if not cell and row_idx > 0 and col_idx < len(expanded[row_idx - 1]):
                cell = expanded[row_idx - 1][col_idx]
            expanded_row.append(cell)
        expanded.append(expanded_row)
    # 合并多层为单行：用 "/" 连接去重后的层级
    result: list[str] = []
    for col_idx in range(target_width):
        parts: list[str] = []
        seen: set[str] = set()
        for row_idx in range(len(expanded)):
            if col_idx < len(expanded[row_idx]):
                cell = expanded[row_idx][col_idx]
                if cell and cell not in seen:
                    parts.append(cell)
                    seen.add(cell)
        result.append("/".join(parts) if parts else "")
    return result


def _median_row_width(rows: list[list[str]]) -> int:
    widths = sorted(len(row) for row in rows if row)
    return widths[len(widths) // 2] if widths else 0


def _keep_layout_block(text: str, two_column: bool, domain: str, config: LayoutParseConfig) -> bool:
    compact = compact_for_search(text)
    if len(compact) < config.min_block_chars or PAGE_NUMBER_RE.fullmatch(compact):
        return False
    numeric_count = len(NUMBER_RE.findall(compact))
    high_signal = any(term in compact for term in FINANCIAL_TERMS)
    if domain == "financial_reports":
        return two_column or high_signal or numeric_count >= 2
    if domain == "research":
        return two_column or numeric_count >= 2
    return two_column or (high_signal and numeric_count >= 1)


def _split_block(text: str, max_chars: int) -> list[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    output: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split = max(text.rfind(mark, start + max_chars // 2, end) for mark in ("。", "；", ";", "\n"))
            if split > start:
                end = split + 1
        output.append(text[start:end])
        start = end
    return output


def _infer_block_section(
    lines: list[VisualLine],
    bbox: tuple[float, float, float, float],
) -> str:
    candidates = [line for line in lines if line.bbox[3] <= bbox[1] + 2 and bbox[1] - line.bbox[3] <= 140]
    for line in reversed(candidates):
        if _looks_like_caption(line):
            return compact_for_search(line.text)
    return ""


def _make_chunk(
    document: Document,
    *,
    page: int,
    body: str,
    chunk_type: str,
    index: int,
    section: str,
    tables: list[str],
    metadata: dict[str, Any],
) -> Chunk:
    digest = hashlib.sha1(f"{document.doc_id}:{page}:{chunk_type}:{body}".encode("utf-8")).hexdigest()[:12]
    chunk = Chunk(
        chunk_id=f"layout:{digest}:p{page}:{index}",
        doc_id=document.doc_id,
        domain=document.domain,
        page=page,
        section=section,
        clause_id="",
        text=body,
        tables=tables,
        numbers=extract_numbers(body),
        dates=extract_dates(body),
        metadata={
            "title": document.title,
            "path": document.path,
            "chunk_type": chunk_type,
            "hierarchy_level": "child",
            "parser_name": "pymupdf_layout_v14",
            **metadata,
        },
    )
    chunk.metadata["extra_index_fields"] = build_extra_index_fields(chunk)
    return chunk


def _append_unique(output: list[Chunk], chunk: Chunk, seen: set[str]) -> None:
    key = re.sub(r"\s+", "", chunk.text).casefold()
    if not key or key in seen:
        return
    seen.add(key)
    output.append(chunk)


def _is_margin_text(
    text: str,
    bbox: tuple[float, float, float, float],
    page_height: float,
    recurring: set[str],
    config: LayoutParseConfig,
) -> bool:
    signature = _margin_signature(text)
    near_margin = bbox[3] <= page_height * config.margin_ratio or bbox[1] >= page_height * (1 - config.margin_ratio)
    return bool(near_margin and (signature in recurring or PAGE_NUMBER_RE.fullmatch(compact_for_search(text))))


def _margin_signature(text: str) -> str:
    compact = re.sub(r"\s+", "", normalize_text(text)).casefold()
    compact = re.sub(r"\d+", "#", compact)
    return compact if len(compact) >= 2 else ""


def _clean_cell(value: str) -> str:
    value = compact_for_search(value)
    # PDF 常把中文单元格换行后插入空格；中文之间的空格不承载列边界。
    return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)


def _stable_table_id(path: Path, page: int, index: int, table: LayoutTable) -> str:
    seed = f"{path.name}:{page}:{index}:{table.caption}:{table.header}:{table.bbox}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:14]


def _union_bbox(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    values = list(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _bbox_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def _horizontal_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    width = max(1.0, min(first[2] - first[0], second[2] - second[0]))
    return intersection / width
