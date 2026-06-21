"""Claim-level sparse retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass

from agent.retrieve.claims import ClaimTarget
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import Question, RetrievalResult


@dataclass(frozen=True)
class ClaimQueryBundle:
    query: str
    intent: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"query": self.query, "intent": self.intent, "weight": self.weight}



def build_claim_query_bundles(question: Question, claim: ClaimTarget, *, max_bundles: int = 6) -> list[ClaimQueryBundle]:
    bundles: list[ClaimQueryBundle] = []
    bundles.append(ClaimQueryBundle(_join(question.question, claim.option_key, claim.option_text), "stem_claim", 1.0))
    if claim.entities:
        bundles.append(ClaimQueryBundle(_join(*claim.entities[:8]), "entity_anchor", 0.9))
    if claim.claim_type in {"metric_fact", "comparison"}:
        bundles.append(ClaimQueryBundle(_join(question.question, claim.option_text, *claim.numbers[:4]), "metric_value", 1.0))
    if claim.claim_type == "comparison":
        bundles.append(ClaimQueryBundle(_join(question.question, claim.option_text, "比较", "数值", "同比", "高于", "低于"), "comparison_endpoint", 0.95))
    if claim.claim_type == "date_fact":
        bundles.append(ClaimQueryBundle(_join(question.question, claim.option_text, *claim.dates[:4], "日期", "期限", "时间"), "date_schedule", 0.95))
    if claim.claim_type == "clause_consequence":
        bundles.append(ClaimQueryBundle(_join(question.question, claim.option_text, "处罚", "扣减", "期限", "不得", "应当"), "clause_consequence", 1.0))
    if claim.should_terms:
        bundles.append(ClaimQueryBundle(_join(*claim.must_terms[:4], *claim.should_terms[:4]), "claim_terms", 0.75))
    return _dedupe_bundles(bundles)[:max_bundles]



def retrieve_claim_candidates(
    index,
    question: Question,
    claim: ClaimTarget,
    *,
    filter_doc_ids: set[str] | None,
    top_k_per_query: int,
    fused_top_k: int,
    shared_candidates: list[RetrievalResult] | None = None,
) -> tuple[list[RetrievalResult], list[ClaimQueryBundle]]:
    bundles = build_claim_query_bundles(question, claim)
    ranked_lists = [
        index.search(
            query=bundle.query,
            top_k=top_k_per_query,
            filter_doc_ids=filter_doc_ids,
            source=f"claim_{claim.option_key}_{bundle.intent}",
        )
        for bundle in bundles
    ]
    if shared_candidates:
        ranked_lists.append(shared_candidates)
    return reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k), bundles



def _join(*parts) -> str:
    return " ".join(str(part or "").strip() for part in parts if str(part or "").strip())



def _dedupe_bundles(bundles: list[ClaimQueryBundle]) -> list[ClaimQueryBundle]:
    output: list[ClaimQueryBundle] = []
    seen: set[str] = set()
    for bundle in bundles:
        normalized = " ".join(bundle.query.split())
        if normalized and normalized not in seen:
            output.append(ClaimQueryBundle(normalized, bundle.intent, bundle.weight))
            seen.add(normalized)
    return output
