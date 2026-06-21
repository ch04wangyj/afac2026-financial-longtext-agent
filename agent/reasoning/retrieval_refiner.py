"""Adaptive retrieval target refinement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.reasoning.answer_parser import extract_json_object


@dataclass
class RetrievalRefinementResult:
    goal: str = ""
    search_intent: str = ""
    refined_queries: list[str] = field(default_factory=list)
    keep_terms: list[str] = field(default_factory=list)
    avoid_terms: list[str] = field(default_factory=list)


@dataclass
class LogicRAGQueryBundle:
    query: str
    intent: str = ""
    evidence_type: str = ""
    must_terms: list[str] = field(default_factory=list)
    doc_scope_hint: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "evidence_type": self.evidence_type,
            "must_terms": list(self.must_terms),
            "doc_scope_hint": list(self.doc_scope_hint),
        }


@dataclass
class LogicRAGSufficiencyJudgement:
    sufficient: bool = False
    failure_tags: list[str] = field(default_factory=list)
    reason: str = ""
    missing_evidence: str = ""
    next_search_goal: str = ""

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "failure_tags": list(self.failure_tags),
            "reason": self.reason,
            "missing_evidence": self.missing_evidence,
            "next_search_goal": self.next_search_goal,
        }


def should_trigger_retrieval_refinement(*, domain: str, sufficiency: dict) -> tuple[bool, str]:
    failure_tags = list(sufficiency.get("failure_tags") or [])
    if sufficiency.get("sufficient"):
        return False, ""
    priority = [
        "missing_metric_value_pair",
        "same_doc_wrong_clause",
        "missing_clause_consequence",
        "generic_context_only",
        "missing_second_endpoint",
    ]
    for tag in priority:
        if tag in failure_tags:
            return True, tag
    return False, ""


def parse_retrieval_refinement_result(raw_text: str) -> RetrievalRefinementResult:
    obj = extract_json_object(raw_text) or {}
    return RetrievalRefinementResult(
        goal=" ".join(str(obj.get("goal", "")).split()),
        search_intent=" ".join(str(obj.get("search_intent", "")).split()),
        refined_queries=_dedupe([str(item) for item in (obj.get("refined_queries") or [])])[:6],
        keep_terms=_dedupe([str(item) for item in (obj.get("keep_terms") or [])])[:8],
        avoid_terms=_dedupe([str(item) for item in (obj.get("avoid_terms") or [])])[:8],
    )


def parse_logicrag_query_bundles(raw_text: str, max_bundles: int = 4) -> list[LogicRAGQueryBundle]:
    obj = extract_json_object(raw_text) or {}
    rows = obj.get("query_bundles") or obj.get("queries") or []
    output: list[LogicRAGQueryBundle] = []
    seen: set[str] = set()
    for row in rows:
        data = {} if isinstance(row, str) else dict(row or {})
        query = " ".join((row if isinstance(row, str) else str(data.get("query", ""))).split())
        if not query or query in seen:
            continue
        seen.add(query)
        output.append(
            LogicRAGQueryBundle(
                query=query,
                intent=" ".join(str(data.get("intent", "")).split()),
                evidence_type=" ".join(str(data.get("evidence_type", "")).split()),
                must_terms=_dedupe([str(item) for item in data.get("must_terms", [])])[:8],
                doc_scope_hint=_dedupe([str(item) for item in data.get("doc_scope_hint", [])])[:8],
            )
        )
        if len(output) >= max_bundles:
            break
    return output


def parse_logicrag_sufficiency_judgement(raw_text: str) -> LogicRAGSufficiencyJudgement:
    obj = extract_json_object(raw_text) or {}
    raw_sufficient = obj.get("sufficient", obj.get("enough", 0))
    sufficient = raw_sufficient is True or str(raw_sufficient).strip() == "1"
    return LogicRAGSufficiencyJudgement(
        sufficient=sufficient,
        failure_tags=_dedupe([str(item) for item in obj.get("failure_tags", [])])[:6],
        reason=" ".join(str(obj.get("reason", "")).split())[:120],
        missing_evidence=" ".join(str(obj.get("missing_evidence", "")).split())[:160],
        next_search_goal=" ".join(str(obj.get("next_search_goal", "")).split())[:160],
    )


def build_lightweight_refined_queries(
    *,
    question_text: str,
    option_key: str,
    option_text: str,
    sufficiency: dict,
    prior_queries: list[str],
) -> RetrievalRefinementResult:
    trigger, reason = should_trigger_retrieval_refinement(domain="", sufficiency=sufficiency)
    base = [question_text, option_key, option_text]
    refined_queries = list(prior_queries)
    goal = ""
    search_intent = ""

    if trigger and reason == "missing_metric_value_pair":
        goal = "改为寻找可直接比较的双边指标值块"
        search_intent = "find_metric_value_block"
        refined_queries.append(f"{question_text} {option_text} 指标 数值 单位")
        refined_queries.append(f"{option_text} 同比 数值")
    elif trigger and reason == "same_doc_wrong_clause":
        goal = "改为寻找同一法规中的具体后果条款"
        search_intent = "find_clause_consequence"
        refined_queries.append(f"{question_text} 扣减 处罚 条款")
    elif trigger and reason == "missing_clause_consequence":
        goal = "改为寻找具体处罚或期限条款"
        search_intent = "find_clause_consequence"
        refined_queries.append(f"{question_text} 处罚 扣减 期限")
    elif trigger and reason == "generic_context_only":
        goal = "避开泛背景页，转向具体事实块"
        search_intent = "avoid_generic_context"
        refined_queries.append(f"{question_text} 具体数值 具体条款")
    elif trigger and reason == "missing_second_endpoint":
        goal = "补齐比较所缺失的另一端点证据"
        search_intent = "complete_comparison_endpoint"
        refined_queries.append(f"{question_text} 另一方 数值 日期")

    return RetrievalRefinementResult(
        goal=goal,
        search_intent=search_intent,
        refined_queries=_dedupe([query for query in refined_queries if query] + [" ".join(base).strip()])[:6],
        keep_terms=[],
        avoid_terms=[],
    )


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join((item or "").split())
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
