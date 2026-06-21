"""Claim 级稀疏检索与金融指标查询扩展。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.retrieve.claims import ClaimTarget
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import Question, RetrievalResult


_MULTI_DOC_HINTS = ("均", "都", "双方", "两份", "两家", "各自", "所有", "分别", "高于", "低于", "快于", "慢于")
_FINANCIAL_METRIC_ALIASES = (
    (
        ("归母净利润", "净利润增速", "归属于母公司"),
        ("归属于上市公司股东的净利润", "母公司拥有人应占溢利", "本年比上年增减"),
    ),
    (
        ("研发投入", "研发费用", "研发强度"),
        ("研发投入占营业收入比例", "研发费用占营业收入比例", "研发投入金额"),
    ),
    (
        ("每股现金分红", "每股分红", "现金分红", "每股派息"),
        ("每10股派发现金红利", "每10股派息数", "现金分红金额", "利润分配预案"),
    ),
    (
        ("经营活动现金流", "经营现金流"),
        ("经营活动产生的现金流量净额", "经营活动现金流净额"),
    ),
    (
        ("营业收入", "营业额"),
        ("营业收入", "营业额", "本年比上年增减"),
    ),
)


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
        metric_aliases = _financial_metric_aliases(claim)
        if metric_aliases:
            bundles.append(
                ClaimQueryBundle(
                    _join(*metric_aliases, *claim.dates[:2], *claim.numbers[:2]),
                    "metric_alias",
                    3.0,
                )
            )
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
    ranked_weights = [bundle.weight for bundle in bundles]
    # 多文档比较不能只依赖一个混合候选池，否则词频更高的文档会占满 Top-K。
    # 对每份指定文档重复相同的 claim 查询，使每个比较端点都有独立召回机会。
    if _requires_doc_scoped_search(claim, filter_doc_ids):
        for doc_index, doc_id in enumerate(sorted(filter_doc_ids or ())):
            for bundle in bundles:
                ranked_lists.append(
                    index.search(
                        query=bundle.query,
                        top_k=top_k_per_query,
                        filter_doc_ids={doc_id},
                        source=f"claim_{claim.option_key}_{bundle.intent}_doc{doc_index + 1}",
                    )
                )
                ranked_weights.append(bundle.weight)
    if shared_candidates:
        ranked_lists.append(shared_candidates)
        ranked_weights.append(0.5)
    fused = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k, weights=ranked_weights)
    metric_aliases = _financial_metric_aliases(claim)
    if metric_aliases:
        for item in fused:
            item.metadata["claim_metric_alias_hits"] = [
                alias for alias in metric_aliases if alias in (item.evidence_text or "")
            ]
    return fused, bundles


def _requires_doc_scoped_search(claim: ClaimTarget, filter_doc_ids: set[str] | None) -> bool:
    if not filter_doc_ids or len(filter_doc_ids) < 2:
        return False
    return claim.claim_type == "comparison" or any(hint in claim.option_text for hint in _MULTI_DOC_HINTS)


def _financial_metric_aliases(claim: ClaimTarget) -> list[str]:
    """把题面简称扩为财报常见披露名，保持查询可审计且不引入模型。"""
    text = f"{claim.source_question} {claim.option_text}"
    aliases: list[str] = []
    for triggers, canonical_terms in _FINANCIAL_METRIC_ALIASES:
        if any(trigger in text for trigger in triggers):
            aliases.extend(canonical_terms)
    return list(dict.fromkeys(aliases))



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
