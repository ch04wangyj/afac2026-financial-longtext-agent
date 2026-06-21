"""从财报页面文本中提取可检索、可计算的指标行。"""

from __future__ import annotations

import re


_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("归属于上市公司股东的净利润", ("归属于上市公司股东的净利润", "归母净利润", "母公司拥有人应占溢利")),
    ("经营活动产生的现金流量净额", ("经营活动产生的现金流量净额", "经营活动现金流净额", "经营现金流净额")),
    ("研发投入占营业收入比例", ("研发投入占营业收入比例", "研发费用占营业收入比例", "研发投入强度")),
    ("研发投入", ("研发投入金额", "研发投入", "研发费用")),
    ("营业收入", ("营业收入", "营业额")),
    ("净利润", ("净利润",)),
    ("每股收益", ("基本每股收益", "稀释每股收益", "每股收益")),
    ("每10股现金分红", ("每10股派息数", "每10股派发现金红利", "每10股派发现金红利人民币")),
    ("每股现金分红", ("每股现金分红", "每股派息")),
    ("现金分红金额", ("现金分红金额", "现金红利总额")),
    ("资产负债率", ("资产负债率",)),
    ("总资产", ("总资产",)),
    ("归属于上市公司股东的净资产", ("归属于上市公司股东的净资产", "归属于母公司股东权益")),
)
_ORDERED_ALIASES = sorted(
    ((alias, canonical) for canonical, aliases in _METRIC_ALIASES for alias in aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)
_VALUE_RE = re.compile(
    r"(?P<paren>[(（])?\s*(?P<number>[-+]?\d[\d,，]*(?:\.\d+)?)\s*"
    r"(?(paren)[)）])\s*(?P<unit>元/股|%|％|元|千元|百万元|万元|亿元|万|亿|倍)?"
)
_HEADER_LABEL_RE = re.compile(
    r"20\d{2}\s*年(?:末|度)?|本年(?:末)?比上年(?:末)?增减|同比(?:增长|下降)?|变动比例"
)
_UNIT_RE = re.compile(r"(?:单位|币种)\s*[:：]?\s*(?:人民币)?\s*(元/股|元|千元|百万元|万元|亿元)")
_INLINE_UNIT_RE = re.compile(r"[（(]\s*(元/股|元|千元|百万元|万元|亿元|%|％)\s*[）)]")
_YEAR_RE = re.compile(r"20\d{2}")


def extract_financial_metric_rows(text: str, *, max_rows: int = 48, default_year: str = "") -> list[dict]:
    """提取指标、表头、单位和单元格；结果可直接写入 Chunk.metadata。"""
    lines = [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]
    rows: list[dict] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for index, line in enumerate(lines):
        metric_match = _find_metric(line)
        if metric_match is None:
            continue
        alias, canonical, alias_start, alias_end = metric_match
        row_text = _truncate_after_metric_sentence(line, alias_end)
        values = _extract_values(row_text[alias_end:])
        lookahead = index + 1
        while not values and lookahead < len(lines) and lookahead <= index + 2:
            next_line = lines[lookahead]
            if _find_metric(next_line) is not None:
                break
            row_text = f"{row_text} {next_line}"
            values = _extract_values(row_text[alias_end:])
            lookahead += 1
        if not values:
            continue

        header = "" if _looks_narrative(row_text) else _nearest_header(lines, index)
        labels = _HEADER_LABEL_RE.findall(header)
        unit = _infer_unit(row_text, lines, index)
        row_years = list(dict.fromkeys(_YEAR_RE.findall(f"{header} {row_text[:alias_start]}")))
        if not row_years and default_year:
            row_years = [default_year]
        cells = _build_cells(values, labels, row_years, unit)
        if not cells:
            continue
        key = (canonical, header, tuple(cell["raw_value"] for cell in cells))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "metric": canonical,
                "raw_metric": alias,
                "header": header,
                "unit": unit or "未标明",
                "raw_row": row_text,
                "cells": cells,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def format_financial_metric_row(row: dict, *, title: str = "") -> str:
    """把结构化指标行格式化为紧凑、保留原值的检索文本。"""
    parts = [part for part in (title, f"财务指标: {row.get('metric', '')}") if part]
    if row.get("header"):
        parts.append(f"表头: {row['header']}")
    if row.get("unit"):
        parts.append(f"单位: {row['unit']}")
    parts.append(f"原始行: {row.get('raw_row', '')}")
    return "\n".join(parts)


def _find_metric(line: str) -> tuple[str, str, int, int] | None:
    best: tuple[str, str, int, int] | None = None
    for alias, canonical in _ORDERED_ALIASES:
        match = re.search(r"\s*".join(re.escape(char) for char in alias), line)
        if match is None:
            continue
        candidate = (alias, canonical, match.start(), match.end())
        if best is None or match.start() < best[2] or (match.start() == best[2] and len(alias) > len(best[0])):
            best = candidate
    return best


def _extract_values(segment: str) -> list[dict]:
    values: list[dict] = []
    for match in _VALUE_RE.finditer(segment):
        number = match.group("number")
        unit = match.group("unit") or ""
        if not unit and re.fullmatch(r"20\d{2}", number.replace(",", "")):
            continue
        negative = bool(match.group("paren")) and not number.startswith("-")
        raw = f"({number})" if negative else number
        values.append({"raw_value": raw, "unit": unit})
    return values


def _nearest_header(lines: list[str], index: int) -> str:
    for prior in reversed(lines[max(0, index - 8) : index]):
        labels = _HEADER_LABEL_RE.findall(prior)
        years = set(_YEAR_RE.findall(" ".join(labels)))
        if len(years) >= 2 or any(marker in prior for marker in ("增减", "变动比例")):
            return prior
    return ""


def _infer_unit(row_text: str, lines: list[str], index: int) -> str:
    inline = _INLINE_UNIT_RE.search(row_text)
    if inline:
        return inline.group(1)
    for prior in reversed(lines[max(0, index - 10) : index]):
        match = _UNIT_RE.search(prior)
        if match:
            return match.group(1)
    return ""


def _looks_narrative(row_text: str) -> bool:
    return any(
        marker in row_text
        for marker in ("金额为", "约为", "达到", "均实现", "同比上升", "同比下降", "同比增长", "同比减少")
    )


def _truncate_after_metric_sentence(line: str, alias_end: int) -> str:
    """一行包含多个叙述句时，只保留当前指标所在句，避免吸入下一指标。"""
    match = re.search(r"[。；;]", line[alias_end:])
    if match is None:
        return line
    return line[: alias_end + match.start() + 1]


def _build_cells(values: list[dict], labels: list[str], row_years: list[str], row_unit: str) -> list[dict]:
    cells: list[dict] = []
    header_years = [match.group(0) for label in labels if (match := _YEAR_RE.search(label))]
    current_year = header_years[0] if header_years else (row_years[0] if len(row_years) == 1 else "")
    for index, value in enumerate(values):
        label = labels[index] if index < len(labels) else ""
        year_match = _YEAR_RE.search(label)
        is_change_column = any(marker in label for marker in ("增减", "同比", "变动"))
        year = year_match.group(0) if year_match else current_year if is_change_column else (row_years[0] if len(row_years) == 1 else "")
        cells.append(
            {
                "column": label,
                "year": year,
                "raw_value": value["raw_value"],
                "unit": value["unit"] or row_unit or "未标明",
            }
        )
    return cells
