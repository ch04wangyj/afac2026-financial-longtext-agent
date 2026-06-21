"""从证据编译可审计的数值事实账本。

账本只做确定性抽取与单位规范化，不猜测缺失指标，也不执行模型生成代码。
它用于把长文本中的数值压缩成最终复核可直接核对的结构化事实。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

from agent.schemas import Question, RetrievalResult


_VALUE_RE = re.compile(
    r"(?P<paren>[(（])?\s*(?P<value>[-+]?\d[\d,，]*(?:\.\d+)?)\s*"
    r"(?(paren)[)）])\s*(?P<unit>%|％|元|千元|百万元|万元|亿元|万|亿|倍)?"
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_HEADER_UNIT_RE = re.compile(r"(?:单位|币种)\s*[:：]?\s*(人民币)?\s*(元|千元|万元|亿元|万|亿)")
_METRICS = (
    "经营活动产生的现金流量净额",
    "经营活动现金流净额",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "营业收入",
    "净利润",
    "研发投入",
    "现金分红",
    "资产负债率",
    "每股收益",
    "发行规模",
    "票面利率",
    "保险金额",
    "赔付金额",
    "赔偿限额",
    "免赔额",
    "保费",
)
_UNIT_MULTIPLIERS = {
    "元": Decimal("1"),
    "千元": Decimal("1000"),
    "百万元": Decimal("1000000"),
    "万元": Decimal("10000"),
    "万": Decimal("10000"),
    "亿元": Decimal("100000000"),
    "亿": Decimal("100000000"),
    "%": Decimal("0.01"),
    "％": Decimal("0.01"),
    "倍": Decimal("1"),
}


@dataclass(frozen=True)
class NumericFact:
    fact_id: str
    doc_id: str
    chunk_id: str
    metric: str
    year: str
    raw_value: str
    unit: str
    normalized_value: str
    context: str
    extraction_mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def compile_numeric_fact_ledger(
    question: Question,
    evidence: list[RetrievalResult],
    *,
    max_facts: int = 36,
) -> dict:
    """从 evidence 编译数值事实，并按题目相关性排序去重。"""
    query_text = f"{question.question} {' '.join(question.options.values())}"
    facts: list[tuple[float, NumericFact]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for evidence_index, item in enumerate(evidence, start=1):
        text = " ".join((item.evidence_text or "").split())
        if not text:
            continue
        financial_row = item.metadata.get("financial_row") or {}
        if financial_row:
            metric = str(financial_row.get("metric") or "未识别指标")
            context = str(financial_row.get("raw_row") or text)
            for cell_index, cell in enumerate(financial_row.get("cells", []), start=1):
                raw_number = str(cell.get("raw_value") or "")
                unit = str(cell.get("unit") or financial_row.get("unit") or "")
                negative = raw_number.startswith("(") and raw_number.endswith(")")
                normalized = _normalize_value(raw_number.strip("()（）"), unit, negative=negative)
                if normalized is None:
                    continue
                year = str(cell.get("year") or "")
                key = (item.doc_id, item.chunk_id, metric, year, f"{normalized}:{unit}")
                if key in seen:
                    continue
                seen.add(key)
                fact = NumericFact(
                    fact_id=f"F{evidence_index}_R{cell_index}",
                    doc_id=item.doc_id,
                    chunk_id=item.chunk_id,
                    metric=metric,
                    year=year,
                    raw_value=raw_number,
                    unit=unit or "未标明",
                    normalized_value=_decimal_text(normalized),
                    context=context,
                    extraction_mode="financial_row",
                )
                facts.append((_fact_relevance(fact, query_text, item) + 1.5, fact))
            # 行级 chunk 已提供准确列映射，不再对同一文本做邻近年份猜测。
            continue
        header_unit = _infer_header_unit(text)
        for mention_index, match in enumerate(_VALUE_RE.finditer(text), start=1):
            raw_number = match.group("value")
            suffix_unit = match.group("unit") or ""
            unit = suffix_unit or header_unit
            if _is_plain_year(raw_number, unit):
                continue
            context = _context_window(text, match.start(), match.end())
            metric = _nearest_metric(context, query_text)
            year = _nearest_year(context)
            negative = bool(match.group("paren")) and not raw_number.startswith("-")
            normalized = _normalize_value(raw_number, unit, negative=negative)
            if normalized is None:
                continue
            key = (item.doc_id, item.chunk_id, metric, year, f"{normalized}:{unit}")
            if key in seen:
                continue
            seen.add(key)
            fact = NumericFact(
                fact_id=f"F{evidence_index}_{mention_index}",
                doc_id=item.doc_id,
                chunk_id=item.chunk_id,
                metric=metric,
                year=year,
                raw_value=("(" if negative else "") + raw_number + (")" if negative else ""),
                unit=unit or "未标明",
                normalized_value=_decimal_text(normalized),
                context=context,
                extraction_mode="text_regex",
            )
            facts.append((_fact_relevance(fact, query_text, item), fact))

    ranked = [fact for _, fact in sorted(facts, key=lambda row: (-row[0], row[1].fact_id))[:max_facts]]
    return {
        "facts": [fact.to_dict() for fact in ranked],
        "fact_count": len(ranked),
        "source_doc_ids": list(dict.fromkeys(fact.doc_id for fact in ranked)),
        "missing": [] if ranked else ["numeric_fact"],
    }


def format_numeric_fact_ledger(ledger: dict) -> str:
    """把账本压缩成最终复核 prompt 可读格式。"""
    facts = list(ledger.get("facts") or [])
    if not facts:
        return "无可验证数值事实"
    lines = []
    for fact in facts:
        lines.append(
            "{fact_id} | doc={doc_id} | metric={metric} | year={year} | raw={raw_value} {unit} | "
            "normalized={normalized_value} | mode={extraction_mode} | {context}".format(**fact)
        )
    return "\n".join(lines)


def _infer_header_unit(text: str) -> str:
    match = _HEADER_UNIT_RE.search(text[:500])
    return match.group(2) if match else ""


def _context_window(text: str, start: int, end: int, radius: int = 56) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip()


def _nearest_metric(context: str, query_text: str) -> str:
    candidates = [metric for metric in _METRICS if metric in context]
    if candidates:
        return max(candidates, key=len)
    query_candidates = [metric for metric in _METRICS if metric in query_text]
    return max(query_candidates, key=len) if query_candidates else "未识别指标"


def _nearest_year(context: str) -> str:
    matches = _YEAR_RE.findall(context)
    return matches[-1] if matches else ""


def _normalize_value(raw: str, unit: str, *, negative: bool) -> Decimal | None:
    cleaned = raw.replace(",", "").replace("，", "")
    try:
        value = Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None
    if negative:
        value = -value
    return value * _UNIT_MULTIPLIERS.get(unit, Decimal("1"))


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _is_plain_year(raw: str, unit: str) -> bool:
    return not unit and bool(re.fullmatch(r"(?:19|20)\d{2}", raw.replace(",", "")))


def _fact_relevance(fact: NumericFact, query_text: str, item: RetrievalResult) -> float:
    score = 0.0
    if fact.metric != "未识别指标":
        score += 2.0
        if fact.metric in query_text:
            score += 2.0
    if fact.year and fact.year in query_text:
        score += 1.5
    if fact.unit != "未标明":
        score += 0.8
    if item.metadata.get("chunk_type") in {"table", "figure"}:
        score += 0.8
    score += min(1.0, float(item.score) / 10.0)
    return score
