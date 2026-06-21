"""受支持的 RAG 检索变体集合。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.index.bm25 import BM25SearchIndex
from agent.reasoning.logicrag import build_logicrag_rrf_queries
from agent.retrieve.doc_first import retrieve_doc_first
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.query import build_rule_queries
from agent.retrieve.targets import question_with_options
from agent.schemas import Question, RetrievalResult


@dataclass(frozen=True)
class RagVariant:
    """一个可运行/可评估的检索策略描述。"""

    name: str
    description: str


RAG_VARIANTS = [
    RagVariant(
        "doc_first_bm25f_expansion",
        "Formal default path: BM25F-lite sparse retrieval, doc-first local expansion, and offline expansion-field support",
    ),
    RagVariant(
        "logicrag_qwen_rrf",
        "LogicRAG retrieval-first: Qwen-planned subproblems fused with BM25/RRF",
    ),
]


def get_variant(name: str) -> RagVariant:
    """按名称查找当前保留的检索变体。"""
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
    """根据保留的 variant 名称执行对应检索逻辑。"""
    variant = get_variant(variant_name)

    if variant.name == "doc_first_bm25f_expansion":
        keyword_bundles = [tuple(query.split()) for query in build_rule_queries(question)]
        results = retrieve_doc_first(index, keyword_bundles=keyword_bundles, top_docs=max(8, top_k), top_k=top_k)
        for result in results:
            result.source = variant.name
        return results[:top_k]

    if variant.name == "logicrag_qwen_rrf":
        seed_results = index.search(
            question_with_options(question),
            top_k=top_k,
            filter_doc_ids=set(question.doc_ids) if question.doc_ids else None,
            source=f"{variant.name}:seed",
        )
        ranked_lists = [seed_results]
        ranked_lists.extend(
            index.search(
                query,
                top_k=top_k,
                filter_doc_ids=set(question.doc_ids) if question.doc_ids else None,
                source=variant.name,
            )
            for query in build_logicrag_rrf_queries(question, seed_results=seed_results)
        )
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k)

    raise AssertionError(f"Unhandled variant: {variant.name}")
