"""领域索引增强规则。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agent.schemas import Chunk

FINANCIAL_METRICS = ["营业收入", "净利润", "归属于上市公司股东的净利润", "现金流", "资产负债率", "研发投入", "股份回购"]
TRANSACTION_TERMS = ["交易对方", "发行人", "受托管理人", "认购", "股东", "募集说明书", "发行规模", "评级", "担保"]
INSURANCE_TERMS = ["保险责任", "责任免除", "等待期", "犹豫期", "退保", "身故保险金", "现金价值"]
REGULATORY_TERMS = ["中国证监会", "决定", "公告", "附件", "施行", "受益所有人", "客户尽职调查"]
RESEARCH_TERMS = ["摘要", "投资要点", "结论", "图", "表", "预测", "评级", "市场规模", "渗透率"]
YEAR_UNIT_RE = re.compile(r"\b20\d{2}\s*年\b|\d+(?:\.\d+)?\s*(?:元|万元|亿元|%|％)")


def build_extra_index_fields(chunk: Chunk | Mapping[str, Any]) -> list[str]:
    from agent.retrieve.structured_queries import extract_query_entities

    row = _as_row(chunk)
    domain = str(row.get("domain", ""))
    text = str(row.get("text", ""))
    section = str(row.get("section", ""))
    clause_id = str(row.get("clause_id", ""))
    tables = row.get("tables", []) or []
    metadata = row.get("metadata", {}) or {}
    title = str(metadata.get("title", ""))

    fields: list[str] = []
    if domain == "financial_reports":
        fields.extend(_present_terms(text, FINANCIAL_METRICS))
        fields.extend(YEAR_UNIT_RE.findall(text))
        fields.extend(extract_query_entities(title)[:8])
        fields.extend(extract_query_entities(text)[:8])
        fields.extend(_build_financial_expansion_fields(title=title, section=section, text=text))
        financial_row = metadata.get("financial_row") or {}
        if financial_row:
            fields.extend(
                str(value)
                for value in (
                    financial_row.get("metric"),
                    financial_row.get("raw_metric"),
                    financial_row.get("header"),
                    financial_row.get("unit"),
                )
                if value
            )
            for cell in financial_row.get("cells", []):
                fields.extend(str(cell.get(key, "")) for key in ("column", "year", "raw_value", "unit") if cell.get(key))
    elif domain == "financial_contracts":
        fields.extend(_present_terms(text, TRANSACTION_TERMS))
        fields.extend(_present_terms(section, ["重大资产重组", "募集说明书", "交易概况"]))
        fields.extend(extract_query_entities(title)[:8])
        fields.extend(extract_query_entities(text)[:6])
    elif domain == "insurance":
        fields.extend(_present_terms(text, INSURANCE_TERMS))
        fields.extend(extract_query_entities(title)[:8])
        fields.extend(extract_query_entities(text)[:6])
        if clause_id:
            fields.append(clause_id)
    elif domain == "regulatory":
        fields.extend(_present_terms(text, REGULATORY_TERMS))
        fields.extend(extract_query_entities(title)[:8])
        fields.extend(extract_query_entities(text)[:6])
    elif domain == "research":
        fields.extend(_present_terms(text, RESEARCH_TERMS))
        fields.extend(extract_query_entities(title)[:8])
        fields.extend(extract_query_entities(text)[:6])

    caption = str(metadata.get("caption", ""))
    if caption:
        fields.append(caption)
        fields.extend(extract_query_entities(caption)[:4])
    parser_name = str(metadata.get("parser_name", ""))
    if parser_name:
        fields.append(parser_name)
    for table in tables:
        if isinstance(table, str) and table.strip():
            header = table.splitlines()[0].strip()
            if header:
                fields.append(header)
                fields.extend(extract_query_entities(header)[:4])
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


def _build_financial_expansion_fields(*, title: str, section: str, text: str) -> list[str]:
    expansions: list[str] = []
    company_terms = extract_company_like_terms(title)
    years = YEAR_UNIT_RE.findall(text)
    if "股份回购" in text or "回购计划" in text:
        subject = company_terms[0] if company_terms else "公司"
        year = years[0] if years else ""
        prefix = f"{subject} {year}".strip()
        expansions.append(f"结论重述: {prefix} 连续实施股份回购方案".strip())
        expansions.append(f"问法扩展: {subject} 股份回购计划".strip())
    if "营业收入" in text and ("增长" in text or "同比" in text):
        subject = company_terms[0] if company_terms else "公司"
        expansions.append(f"结论重述: {subject} 营业收入增长情况".strip())
    if "归属于母公司所有者的净利润" in text or "归属于上市公司股东的净利润" in text:
        subject = company_terms[0] if company_terms else "公司"
        expansions.append(f"问法扩展: {subject} 归母净利润".strip())
    if section:
        expansions.append(f"可能回答: {section}")
    return _dedupe([item for item in expansions if item.strip()])


def extract_company_like_terms(title: str) -> list[str]:
    candidates = re.findall(r"[\u4e00-\u9fffA-Za-z]+(?:集团股份有限公司|股份有限公司|集团有限公司|有限公司|集团)", title)
    simplified: list[str] = []
    for item in candidates:
        value = item
        for suffix in ("集团股份有限公司", "股份有限公司", "集团有限公司", "有限公司", "集团"):
            if len(value) > len(suffix) and value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        if value:
            simplified.append(value)
    return _dedupe(simplified)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
