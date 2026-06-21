"""Option-level retrieval query expansion and fusion helpers."""

from __future__ import annotations

from agent.retrieve.claim_retrieval import build_claim_query_bundles
from agent.retrieve.claims import build_claim_targets
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import Question, RetrievalResult



def build_option_queries(question: Question, max_queries_per_option: int = 5) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for claim in build_claim_targets(question):
        bundles = build_claim_query_bundles(question, claim, max_bundles=max_queries_per_option)
        output[claim.option_key] = [bundle.query for bundle in bundles]
    return output


def retrieve_option_candidates(
    index,
    question: Question,
    filter_doc_ids: set[str] | None,
    top_k_per_query: int = 12,
    fused_top_k: int = 12,
) -> dict[str, list[RetrievalResult]]:
    output: dict[str, list[RetrievalResult]] = {}
    for option_key, queries in build_option_queries(question).items():
        ranked_lists = [
            index.search(
                query=query,
                top_k=top_k_per_query,
                filter_doc_ids=filter_doc_ids,
                source=f"option_{option_key}",
            )
            for query in queries
        ]
        output[option_key] = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
    return output
