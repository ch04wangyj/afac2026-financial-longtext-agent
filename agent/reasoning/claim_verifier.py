"""Claim verification helpers shared by single-choice and multi-select solving."""

from __future__ import annotations

from dataclasses import dataclass

from agent.retrieve.claims import ClaimTarget, claim_to_retrieval_target
from agent.retrieve.targets import analyze_evidence_sufficiency


@dataclass(frozen=True)
class ClaimRefinement:
    action: str
    reason: str
    queries: list[str]

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "queries": list(self.queries)}



def analyze_claim_evidence_sufficiency(claim: ClaimTarget, evidence_texts: list[str]) -> dict:
    target = claim_to_retrieval_target(claim)
    report = analyze_evidence_sufficiency(target, evidence_texts)
    return {
        "claim_id": claim.claim_id,
        "option_key": claim.option_key,
        "claim_type": claim.claim_type,
        **report,
    }



def build_claim_refinement(claim: ClaimTarget, sufficiency: dict) -> ClaimRefinement:
    tags = set(sufficiency.get("failure_tags") or [])
    base = f"{claim.source_question} {claim.option_text}".strip()
    queries: list[str] = []
    action = "same_goal_retry"

    if "missing_second_endpoint" in tags:
        action = "find_missing_comparison_endpoint"
        queries.append(f"{base} 比较 对比 数值 日期 另一方")
    if "missing_numeric_value" in tags:
        action = "find_metric_value_block"
        queries.append(f"{base} 指标 数值 金额 比率 表")
    if "missing_date_value" in tags:
        action = "find_date_or_schedule_block"
        queries.append(f"{base} 日期 时间 期限 安排")
    if "missing_clause_consequence" in tags or "same_doc_wrong_clause" in tags:
        action = "find_clause_consequence"
        queries.append(f"{base} 处罚 扣减 减分 期限 不得 应当")
    if "generic_context_only" in tags:
        queries.append(f"{base} 具体 数值 条款 结果 决定")

    if not queries:
        queries.append(base)

    return ClaimRefinement(action=action, reason=",".join(sorted(tags)) or "insufficient", queries=_dedupe(queries))



def assemble_claim_answer(verdicts: dict[str, dict], *, answer_format: str) -> str:
    if answer_format == "multi":
        return "".join(
            key for key in sorted(verdicts)
            if verdicts[key].get("relation") == "support"
        )
    supported = [
        (key, float(value.get("confidence", 0.0) or 0.0))
        for key, value in verdicts.items()
        if value.get("relation") == "support"
    ]
    if supported:
        return sorted(supported, key=lambda row: (-row[1], row[0]))[0][0]
    non_refuted = [
        (key, float(value.get("confidence", 0.0) or 0.0))
        for key, value in verdicts.items()
        if value.get("relation") != "refute"
    ]
    if non_refuted:
        return sorted(non_refuted, key=lambda row: (-row[1], row[0]))[0][0]
    return sorted(verdicts)[0] if verdicts else ""



def should_refine_claim(sufficiency: dict, verdict: dict, *, threshold: float) -> bool:
    if not sufficiency.get("sufficient", False):
        return True
    if verdict.get("relation") == "insufficient":
        return True
    if float(verdict.get("confidence", 0.0) or 0.0) < float(threshold):
        return True
    return False



def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(str(item or "").split())
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
