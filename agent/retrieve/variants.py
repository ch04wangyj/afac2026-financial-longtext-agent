"""可横向评估的 RAG 检索变体集合。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.index.bm25 import BM25SearchIndex
from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.query import build_rule_queries
from agent.retrieve.structured_queries import (
    build_graph_lite_queries,
    build_linear_entity_queries,
    build_logic_queries,
)
from agent.schemas import Question, RetrievalResult


@dataclass(frozen=True)
class RagVariant:
    """一个可运行/可评估的检索策略描述。"""

    name: str
    description: str
    restrict_to_gold_docs: bool = False


RAG_VARIANTS = [
    RagVariant("question_only", "BM25 over question text only"),
    RagVariant("question_options", "BM25 over question plus all options"),
    RagVariant("option_rrf", "RRF over one query per option"),
    RagVariant("rule_multi_rrf", "RRF over rule-generated queries with numbers/dates/options"),
    RagVariant("field_boosted_rrf", "rule_multi_rrf plus clause/number/date/title boosts"),
    RagVariant("logic_lite_rrf", "LogicRAG-lite: query-time subproblem DAG approximated by option/entity subqueries"),
    RagVariant("linear_entity_rrf", "LinearRAG-lite: linear high-signal entity queries"),
    RagVariant("graph_lite_rrf", "GraphRAG-lite: entity co-occurrence pair queries without pre-built graph"),
    RagVariant("crag_lite", "CRAG-lite: question_options with corrective fallback to graph/rule retrieval"),
    RagVariant("oracle_doc_restricted", "A-board upper bound: rule_multi_rrf restricted to provided doc_ids", True),
]


def get_variant(name: str) -> RagVariant:
    """按名称查找检索变体，供脚本和测试复用。"""
    for variant in RAG_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(f"Unknown RAG variant: {name}")


def retrieve_with_variant(
    index: BM25SearchIndex,
    question: Question,
    variant_name: str,
    top_k: int = 30,
) -> list[RetrievalResult]:
    """根据 variant 名称执行对应的检索逻辑。"""
    variant = get_variant(variant_name)
    filter_doc_ids = set(question.doc_ids) if variant.restrict_to_gold_docs and question.doc_ids else None

    if variant.name == "question_only":
        return index.search(question.question, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)

    if variant.name == "question_options":
        query = _question_with_options(question)
        return index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)

    if variant.name == "option_rrf":
        ranked_lists = [
            index.search(f"{question.question} {key} {value}", top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for key, value in sorted(question.options.items())
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    if variant.name in {"rule_multi_rrf", "oracle_doc_restricted"}:
        ranked_lists = [
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for query in build_rule_queries(question)
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    if variant.name == "field_boosted_rrf":
        ranked_lists = [
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for query in build_rule_queries(question)
        ]
        fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k * 2)
        boosted = _field_boost(question, fused)
        return boosted[:top_k]

    if variant.name == "logic_lite_rrf":
        ranked_lists = [
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for query in build_logic_queries(question)
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    if variant.name == "linear_entity_rrf":
        ranked_lists = [
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for query in build_linear_entity_queries(question)
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    if variant.name == "graph_lite_rrf":
        ranked_lists = [
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
            for query in build_graph_lite_queries(question)
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    if variant.name == "crag_lite":
        first_pass = index.search(
            query=_question_with_options(question),
            top_k=top_k,
            filter_doc_ids=filter_doc_ids,
            source=variant.name,
        )
        if _retrieval_is_confident(first_pass):
            return first_pass
        fallback_lists = [
            first_pass,
            *[
                index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
                for query in build_graph_lite_queries(question)[:8]
            ],
            *[
                index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source=variant.name)
                for query in build_rule_queries(question)[:6]
            ],
        ]
        return reciprocal_rank_fusion(fallback_lists, top_k=top_k)

    raise AssertionError(f"Unhandled variant: {variant.name}")


def _question_with_options(question: Question) -> str:
    """拼接题干和选项；当前 A 组代理评估中最稳的默认查询。"""
    return f"{question.question} " + " ".join(
        f"{key} {value}" for key, value in sorted(question.options.items())
    )


def _field_boost(question: Question, results: list[RetrievalResult]) -> list[RetrievalResult]:
    """基于条款号、数字、日期、标题和选项命中做轻量加分。"""
    q_text = _question_with_options(question)
    q_numbers = set(extract_numbers(q_text))
    q_dates = set(extract_dates(q_text))
    option_values = list(question.options.values())

    boosted: list[RetrievalResult] = []
    for result in results:
        score = result.score
        evidence = result.evidence_text
        title = str(result.metadata.get("title", ""))
        if result.metadata.get("clause_id"):
            score += 0.025
        if q_numbers & set(result.metadata.get("numbers", [])):
            score += 0.05
        if q_dates & set(result.metadata.get("dates", [])):
            score += 0.05
        if title and title in q_text:
            score += 0.05
        for option in option_values:
            if option[:12] and option[:12] in evidence:
                score += 0.02
        result.score = float(score)
        result.source = "field_boosted_rrf"
        boosted.append(result)
    return sorted(boosted, key=lambda item: item.score, reverse=True)


def _retrieval_is_confident(results: list[RetrievalResult]) -> bool:
    """CRAG-lite 的置信判定：结果少、分数低或首二名过近都视为不稳。"""
    if len(results) < 5:
        return False
    if results[0].score <= 0:
        return False
    if len(results) > 1 and results[1].score > 0 and results[0].score / results[1].score < 1.15:
        return False
    return True
