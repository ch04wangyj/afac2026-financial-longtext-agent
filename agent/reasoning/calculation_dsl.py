"""基于事实账本的白名单金融计算，不执行任意代码。"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations

from agent.schemas import Question


_COMPARE_HINTS = ("高于", "低于", "大于", "小于", "快于", "慢于", "对比", "比较", "双方", "两家")
_GROWTH_HINTS = ("同比", "增速", "增长", "下降", "增加", "减少", "变化")
_ALLOWED_OPERATIONS = {"compare", "difference", "ratio", "growth_rate"}
_METRIC_QUERY_HINTS = {
    "归属于上市公司股东的净利润": ("归属于上市公司股东的净利润", "归母净利润", "净利润增速"),
    "净利润": ("净利润",),
    "营业收入": ("营业收入", "营业额"),
    "经营活动产生的现金流量净额": ("经营活动产生的现金流量净额", "经营活动现金流净额", "经营现金流"),
    "研发投入占营业收入比例": ("研发投入占营业收入比例", "研发投入占比", "研发投入强度", "研发强度"),
    "研发投入": ("研发投入", "研发费用"),
    "每10股现金分红": ("每10股", "每股现金分红", "每股派息", "现金分红"),
    "每股现金分红": ("每股现金分红", "每股派息"),
    "现金分红金额": ("现金分红", "现金红利"),
}


def evaluate_calculation(ledger: dict, operation: str, operands: list[str]) -> dict:
    """对指定 fact id 执行受限计算，并返回可复核表达式。"""
    if operation not in _ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported calculation operation: {operation}")
    if len(operands) != 2:
        raise ValueError("calculation requires exactly two fact ids")
    facts = {str(fact.get("fact_id")): fact for fact in ledger.get("facts", [])}
    if any(fact_id not in facts for fact_id in operands):
        raise KeyError("calculation references an unknown fact id")
    left = _decimal_value(facts[operands[0]])
    right = _decimal_value(facts[operands[1]])
    if operation in {"ratio", "growth_rate"} and right == 0:
        raise ZeroDivisionError("right operand is zero")

    if operation == "compare":
        value = "gt" if left > right else "lt" if left < right else "eq"
        expression = f"{operands[0]}({left}) compare {operands[1]}({right}) = {value}"
    elif operation == "difference":
        value = left - right
        expression = f"{operands[0]}({left}) - {operands[1]}({right}) = {value}"
    elif operation == "ratio":
        value = left / right
        expression = f"{operands[0]}({left}) / {operands[1]}({right}) = {value}"
    else:
        value = (left - right) / abs(right)
        expression = f"({operands[0]}({left}) - {operands[1]}({right})) / abs({operands[1]}) = {value}"
    return {
        "operation": operation,
        "operands": list(operands),
        "result": str(value),
        "expression": expression,
    }


def build_candidate_calculations(question: Question, ledger: dict, *, max_calculations: int = 24) -> list[dict]:
    """按题干风险词生成跨主体比较和同主体同比候选计算。"""
    question_text = f"{question.question} {' '.join(question.options.values())}"
    facts = [
        fact
        for fact in ledger.get("facts", [])
        if fact.get("metric")
        and fact.get("year")
        and fact.get("extraction_mode") == "financial_row"
        and fact.get("unit") not in {"", "未标明"}
        and _metric_relevant(str(fact.get("metric")), question_text)
    ]
    by_metric_year: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    by_doc_metric: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for fact in facts:
        family = _unit_family(str(fact.get("unit") or ""))
        by_metric_year[(str(fact["metric"]), str(fact["year"]), family)].append(fact)
        by_doc_metric[(str(fact["doc_id"]), str(fact["metric"]), family)].append(fact)

    calculations: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    if any(hint in question_text for hint in _COMPARE_HINTS):
        for group in by_metric_year.values():
            best_by_doc: dict[str, dict] = {}
            for fact in group:
                best_by_doc.setdefault(str(fact.get("doc_id", "")), fact)
            for left, right in combinations(best_by_doc.values(), 2):
                if left.get("doc_id") == right.get("doc_id"):
                    continue
                _append_calculation(calculations, seen, ledger, "compare", [left["fact_id"], right["fact_id"]])
                if len(calculations) >= max_calculations:
                    return calculations

    if any(hint in question_text for hint in _GROWTH_HINTS):
        for group in by_doc_metric.values():
            year_facts = sorted(group, key=lambda fact: str(fact.get("year", "")), reverse=True)
            distinct_year_facts: list[dict] = []
            seen_years: set[str] = set()
            for fact in year_facts:
                year = str(fact.get("year", ""))
                if year and year not in seen_years:
                    distinct_year_facts.append(fact)
                    seen_years.add(year)
            if len(distinct_year_facts) < 2:
                continue
            _append_calculation(
                calculations,
                seen,
                ledger,
                "growth_rate",
                [distinct_year_facts[0]["fact_id"], distinct_year_facts[1]["fact_id"]],
            )
            if len(calculations) >= max_calculations:
                return calculations
    return calculations


def _append_calculation(
    output: list[dict],
    seen: set[tuple[str, tuple[str, ...]]],
    ledger: dict,
    operation: str,
    operands: list[str],
) -> None:
    key = (operation, tuple(operands))
    if key in seen:
        return
    seen.add(key)
    try:
        output.append(evaluate_calculation(ledger, operation, operands))
    except (InvalidOperation, KeyError, ValueError, ZeroDivisionError):
        return


def _decimal_value(fact: dict) -> Decimal:
    try:
        return Decimal(str(fact["normalized_value"]))
    except (InvalidOperation, KeyError) as exc:
        raise InvalidOperation(f"invalid normalized fact value: {fact!r}") from exc


def _unit_family(unit: str) -> str:
    if unit in {"%", "％"}:
        return "ratio"
    if unit == "倍":
        return "multiple"
    if unit == "元/股":
        return "per_share"
    if unit in {"元", "千元", "百万元", "万元", "亿元", "万", "亿"}:
        return "money"
    return unit or "scalar"


def _metric_relevant(metric: str, question_text: str) -> bool:
    hints = _METRIC_QUERY_HINTS.get(metric, (metric,))
    return any(hint in question_text for hint in hints)
