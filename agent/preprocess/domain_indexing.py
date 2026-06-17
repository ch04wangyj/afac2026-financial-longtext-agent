"""领域索引增强规则。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agent.schemas import Chunk

FINANCIAL_METRICS = ["营业收入", "净利润", "归属于上市公司股东的净利润", "现金流", "资产负债率"]
TRANSACTION_TERMS = ["交易对方", "发行人", "受托管理人", "认购", "股东", "募集说明书"]
INSURANCE_TERMS = ["保险责任", "责任免除", "等待期", "犹豫期", "退保"]
REGULATORY_TERMS = ["中国证监会", "决定", "公告", "附件", "施行"]
RESEARCH_TERMS = ["摘要", "投资要点", "结论", "图", "表", "预测", "评级"]
YEAR_UNIT_RE = re.compile(r"\b20\d{2}\s*年\b|\d+(?:\.\d+)?\s*(?:元|万元|亿元|%|％)")


def build_extra_index_fields(chunk: Chunk | Mapping[str, Any]) -> list[str]:
    row = _as_row(chunk)
    domain = str(row.get("domain", ""))
    text = str(row.get("text", ""))
    section = str(row.get("section", ""))
    clause_id = str(row.get("clause_id", ""))
    tables = row.get("tables", []) or []
    metadata = row.get("metadata", {}) or {}

    fields: list[str] = []
    if domain == "financial_reports":
        fields.extend(_present_terms(text, FINANCIAL_METRICS))
        fields.extend(YEAR_UNIT_RE.findall(text))
    elif domain == "financial_contracts":
        fields.extend(_present_terms(text, TRANSACTION_TERMS))
        fields.extend(_present_terms(section, ["重大资产重组", "募集说明书", "交易概况"]))
    elif domain == "insurance":
        fields.extend(_present_terms(text, INSURANCE_TERMS))
        if clause_id:
            fields.append(clause_id)
    elif domain == "regulatory":
        fields.extend(_present_terms(text, REGULATORY_TERMS))
    elif domain == "research":
        fields.extend(_present_terms(text, RESEARCH_TERMS))

    caption = str(metadata.get("caption", ""))
    if caption:
        fields.append(caption)
    parser_name = str(metadata.get("parser_name", ""))
    if parser_name:
        fields.append(parser_name)
    for table in tables:
        if isinstance(table, str) and table.strip():
            header = table.splitlines()[0].strip()
            if header:
                fields.append(header)
    return _dedupe(fields)


def _as_row(chunk: Chunk | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, Chunk):
        return {
            "domain": chunk.domain,
            "text": chunk.text,
            "section": chunk.section,
            "clause_id": chunk.clause_id,
            "tables": list(chunk.tables),
            "metadata": dict(chunk.metadata),
        }
    return dict(chunk)


def _present_terms(text: str, candidates: list[str]) -> list[str]:
    return [item for item in candidates if item in text]


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
