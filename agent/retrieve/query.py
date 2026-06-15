"""规则查询扩展。

把题干、选项、数字和日期拆成多条 BM25 查询，用 RRF 进行补召回。
"""

from __future__ import annotations

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.schemas import Question


def build_rule_queries(question: Question) -> list[str]:
    """为一道题构造最多 8 条规则查询。"""
    option_text = " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))
    base = f"{question.question} {option_text}".strip()
    queries = [base]

    numbers = extract_numbers(base)
    dates = extract_dates(base)
    if numbers:
        queries.append(f"{question.question} {' '.join(numbers)}")
    if dates:
        queries.append(f"{question.question} {' '.join(dates)}")
    for key, value in sorted(question.options.items()):
        queries.append(f"{question.question} {key} {value}")

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        # 去重时保留原始顺序，保证主查询优先进入 RRF。
        query = " ".join(query.split())
        if query and query not in seen:
            deduped.append(query)
            seen.add(query)
    return deduped[:8]
