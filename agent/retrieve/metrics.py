"""检索评估指标。

A 榜没有公开标准答案，因此用题目给定 doc_ids 作为文档命中代理标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.schemas import Question, RetrievalResult


@dataclass
class RetrievalMetrics:
    """单题检索命中指标。"""

    qid: str
    domain: str
    variant: str
    tokenizer_mode: str
    gold_doc_count: int
    retrieved_count: int
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    all_gold_at_10: bool
    recall_at_10: float
    mrr_at_10: float
    first_hit_rank: int | None
    retrieved_doc_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为 CSV/JSON 可写入的字典。"""
        return {
            "qid": self.qid,
            "domain": self.domain,
            "variant": self.variant,
            "tokenizer_mode": self.tokenizer_mode,
            "gold_doc_count": self.gold_doc_count,
            "retrieved_count": self.retrieved_count,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "all_gold_at_10": self.all_gold_at_10,
            "recall_at_10": self.recall_at_10,
            "mrr_at_10": self.mrr_at_10,
            "first_hit_rank": self.first_hit_rank,
            "retrieved_doc_ids": self.retrieved_doc_ids,
        }


def evaluate_retrieval(
    question: Question,
    results: list[RetrievalResult],
    variant: str,
    tokenizer_mode: str,
) -> RetrievalMetrics:
    """根据 gold doc_ids 评估一题检索结果。"""
    gold = set(question.doc_ids)
    ranked_docs = _dedupe_doc_ids([result.doc_id for result in results])
    top10 = ranked_docs[:10]
    first_hit_rank = None
    for rank, doc_id in enumerate(ranked_docs[:10], start=1):
        if doc_id in gold:
            first_hit_rank = rank
            break
    found = gold & set(top10)
    return RetrievalMetrics(
        qid=question.qid,
        domain=question.domain,
        variant=variant,
        tokenizer_mode=tokenizer_mode,
        gold_doc_count=len(gold),
        retrieved_count=len(results),
        hit_at_1=_hit_at(ranked_docs, gold, 1),
        hit_at_3=_hit_at(ranked_docs, gold, 3),
        hit_at_5=_hit_at(ranked_docs, gold, 5),
        hit_at_10=_hit_at(ranked_docs, gold, 10),
        all_gold_at_10=bool(gold) and gold.issubset(set(top10)),
        recall_at_10=len(found) / len(gold) if gold else 0.0,
        mrr_at_10=1.0 / first_hit_rank if first_hit_rank else 0.0,
        first_hit_rank=first_hit_rank,
        retrieved_doc_ids=ranked_docs[:10],
    )


def summarize_metrics(rows: list[RetrievalMetrics]) -> list[dict]:
    """按 tokenizer/variant/domain 聚合平均指标。"""
    groups: dict[tuple[str, str, str], list[RetrievalMetrics]] = {}
    for row in rows:
        groups.setdefault((row.tokenizer_mode, row.variant, row.domain), []).append(row)
        groups.setdefault((row.tokenizer_mode, row.variant, "ALL"), []).append(row)

    summary: list[dict] = []
    for (tokenizer_mode, variant, domain), items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "tokenizer_mode": tokenizer_mode,
                "variant": variant,
                "domain": domain,
                "questions": n,
                "hit_at_1": _avg(item.hit_at_1 for item in items),
                "hit_at_3": _avg(item.hit_at_3 for item in items),
                "hit_at_5": _avg(item.hit_at_5 for item in items),
                "hit_at_10": _avg(item.hit_at_10 for item in items),
                "all_gold_at_10": _avg(item.all_gold_at_10 for item in items),
                "recall_at_10": _avg(item.recall_at_10 for item in items),
                "mrr_at_10": _avg(item.mrr_at_10 for item in items),
            }
        )
    return summary


def _dedupe_doc_ids(doc_ids: list[str]) -> list[str]:
    """按检索顺序去重文档 ID，避免同文档多个 chunk 重复计数。"""
    output: list[str] = []
    seen: set[str] = set()
    for doc_id in doc_ids:
        if doc_id not in seen:
            output.append(doc_id)
            seen.add(doc_id)
    return output


def _hit_at(ranked_docs: list[str], gold: set[str], k: int) -> bool:
    """判断 Top-K 文档中是否至少命中一个 gold 文档。"""
    return bool(gold & set(ranked_docs[:k]))


def _avg(values) -> float:
    """对 bool/float 混合指标求平均。"""
    values = list(values)
    if not values:
        return 0.0
    return sum(1.0 if value is True else 0.0 if value is False else float(value) for value in values) / len(values)
