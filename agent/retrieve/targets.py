"""Structured retrieval target builders for LogicRAG sparse retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.retrieve.structured_queries import extract_query_entities
from agent.schemas import Question


@dataclass(frozen=True)
class RetrievalTarget:
    node_id: str
    rank: int
    question: str
    doc_scope: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    option_terms: list[str] = field(default_factory=list)
    evidence_intent: str = "fact"
    query_variants: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "question": self.question,
            "doc_scope": list(self.doc_scope),
            "must_terms": list(self.must_terms),
            "should_terms": list(self.should_terms),
            "numbers": list(self.numbers),
            "dates": list(self.dates),
            "entities": list(self.entities),
            "option_terms": list(self.option_terms),
            "evidence_intent": self.evidence_intent,
            "query_variants": list(self.query_variants),
        }


COMPARE_HINTS = (
    "比较",
    "对比",
    "高于",
    "低于",
    "增加",
    "减少",
    "同比",
    "环比",
    "是否",
)

CLAUSE_CONSEQUENCE_HINTS = (
    "处罚",
    "罚款",
    "扣减",
    "减分",
    "责令",
    "不得",
    "期限",
)

GENERIC_CONTEXT_HINTS = (
    "审计报告",
    "公允反映",
    "坚持依法合规",
    "客观公正",
    "原则",
    "财务报表",
)

SPECIFIC_EVIDENCE_HINTS = (
    "营业收入",
    "净利润",
    "现金流",
    "分红",
    "日期",
    "发行公告",
    "罚款",
    "扣减",
    "减分",
    "责令",
    "期限",
)



def build_retrieval_target(
    question: Question,
    node_text: str,
    *,
    node_id: str = "",
    rank: int = 0,
    prior_memories: list[dict] | None = None,
    doc_scope: list[str] | None = None,
) -> RetrievalTarget:
    question_options = question_with_options(question)
    node_text = " ".join((node_text or question.question).split())
    base_text = " ".join(part for part in [question.question, node_text] if part).strip()
    option_text = " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))

    entities = _dedupe(
        [
            *extract_query_entities(question.question),
            *extract_query_entities(node_text),
        ]
    )
    numbers = _dedupe(extract_numbers(base_text))
    dates = _dedupe(extract_dates(base_text))
    option_terms = _extract_option_terms(question)
    must_terms = _dedupe([*entities[:8], *numbers[:4], *dates[:4]])[:10]
    should_terms = _dedupe([term for term in option_terms if term not in must_terms])[:8]
    memory_anchor = _memory_anchor_text(prior_memories or [])
    query_variants = _build_query_variants(
        question,
        node_text=node_text,
        must_terms=must_terms,
        should_terms=should_terms,
        numbers=numbers,
        dates=dates,
        option_text=option_text,
        memory_anchor=memory_anchor,
    )
    return RetrievalTarget(
        node_id=node_id or f"rank_{rank}",
        rank=rank,
        question=question.question,
        doc_scope=list(doc_scope or question.doc_ids),
        must_terms=must_terms,
        should_terms=should_terms,
        numbers=numbers,
        dates=dates,
        entities=entities[:12],
        option_terms=option_terms,
        evidence_intent=_infer_evidence_intent(base_text, numbers=numbers, dates=dates),
        query_variants=query_variants,
    )



def merge_retrieval_targets(question: Question, targets: list[RetrievalTarget], *, node_id: str, rank: int) -> RetrievalTarget:
    if not targets:
        return build_retrieval_target(question, question.question, node_id=node_id, rank=rank)
    must_terms = _dedupe(term for target in targets for term in target.must_terms)[:12]
    should_terms = _dedupe(term for target in targets for term in target.should_terms if term not in must_terms)[:10]
    numbers = _dedupe(item for target in targets for item in target.numbers)[:6]
    dates = _dedupe(item for target in targets for item in target.dates)[:6]
    entities = _dedupe(item for target in targets for item in target.entities)[:14]
    option_terms = _dedupe(item for target in targets for item in target.option_terms)[:12]
    query_variants = _dedupe(item for target in targets for item in target.query_variants)[:8]
    joined_text = " ".join(_dedupe(target.question for target in targets))
    evidence_intent = _pick_group_intent(targets)
    doc_scope = _dedupe(doc_id for target in targets for doc_id in target.doc_scope)
    return RetrievalTarget(
        node_id=node_id,
        rank=rank,
        question=joined_text or question.question,
        doc_scope=doc_scope,
        must_terms=must_terms,
        should_terms=should_terms,
        numbers=numbers,
        dates=dates,
        entities=entities,
        option_terms=option_terms,
        evidence_intent=evidence_intent,
        query_variants=query_variants,
    )



def question_with_options(question: Question) -> str:
    return f"{question.question} " + " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))



def analyze_evidence_sufficiency(target: RetrievalTarget, evidence_texts: list[str]) -> dict:
    normalized_texts = [" ".join(str(text or "").split()) for text in evidence_texts if str(text or "").strip()]
    combined = "\n".join(normalized_texts)
    found_numbers = [item for item in target.numbers if item and item in combined]
    found_dates = [item for item in target.dates if item and item in combined]
    found_entities = [item for item in target.entities if item and item in combined]
    found_must_terms = [item for item in target.must_terms if item and item in combined]

    missing_numbers = [item for item in target.numbers if item not in found_numbers]
    missing_dates = [item for item in target.dates if item not in found_dates]
    missing_entities = [item for item in target.entities if item not in found_entities]
    missing_must_terms = [item for item in target.must_terms if item not in found_must_terms]

    comparison_incomplete = False
    comparison_closure = 1.0
    failure_tags: list[str] = []

    if target.evidence_intent == "comparison":
        number_like = [*target.numbers, *target.dates]
        found_number_like = [*found_numbers, *found_dates]
        if len(number_like) >= 2:
            comparison_closure = min(1.0, len(found_number_like) / len(number_like))
            comparison_incomplete = len(found_number_like) < 2
            if comparison_incomplete:
                failure_tags.append("missing_second_endpoint")
        elif len(number_like) == 1 and len(found_number_like) < 1:
            comparison_incomplete = True
            comparison_closure = 0.0
            failure_tags.append("missing_numeric_value")

    generic_context_only = _is_generic_context_only(normalized_texts)
    if generic_context_only:
        failure_tags.append("generic_context_only")

    missing_clause_consequence = _looks_like_clause_consequence_question(target.question) and not _has_clause_consequence(normalized_texts)
    if missing_clause_consequence:
        failure_tags.append("missing_clause_consequence")

    if missing_numbers and target.evidence_intent in {"comparison", "number"} and "missing_numeric_value" not in failure_tags:
        failure_tags.append("missing_numeric_value")

    sufficient = not comparison_incomplete and not generic_context_only and not missing_clause_consequence
    if target.entities and len(found_entities) < min(2, len(target.entities)) and target.evidence_intent == "comparison":
        sufficient = False
    if missing_numbers and target.evidence_intent in {"comparison", "number"}:
        sufficient = False

    evidence_density = _estimate_evidence_density(normalized_texts)

    return {
        "sufficient": sufficient,
        "comparison_incomplete": comparison_incomplete,
        "comparison_closure": round(comparison_closure, 6),
        "failure_tags": _dedupe(failure_tags),
        "evidence_density": round(evidence_density, 6),
        "generic_context_only": generic_context_only,
        "missing_clause_consequence": missing_clause_consequence,
        "missing_numbers": missing_numbers,
        "missing_dates": missing_dates,
        "missing_entities": missing_entities,
        "missing_must_terms": missing_must_terms,
        "found_numbers": found_numbers,
        "found_dates": found_dates,
        "found_entities": found_entities,
        "found_must_terms": found_must_terms,
    }



def _extract_option_terms(question: Question) -> list[str]:
    output: list[str] = []
    for _key, value in sorted(question.options.items()):
        compact = " ".join(value.split())
        if 2 <= len(compact) <= 24:
            output.append(compact)
        output.extend(extract_query_entities(value)[:4])
    return _dedupe(output)[:12]



def _build_query_variants(
    question: Question,
    *,
    node_text: str,
    must_terms: list[str],
    should_terms: list[str],
    numbers: list[str],
    dates: list[str],
    option_text: str,
    memory_anchor: str,
) -> list[str]:
    question_options = question_with_options(question)
    queries = [question_options, f"{question.question} {node_text}".strip()]
    if option_text:
        queries.append(f"{node_text} {option_text}".strip())
    if must_terms:
        queries.append(" ".join(must_terms[:8]))
    if should_terms:
        queries.append(" ".join([*must_terms[:4], *should_terms[:4]]).strip())
    if numbers:
        queries.append(f"{question.question} {' '.join(numbers[:4])}".strip())
    if dates:
        queries.append(f"{question.question} {' '.join(dates[:4])}".strip())
    if memory_anchor:
        queries.append(f"{question_options} {memory_anchor}".strip())
    return _dedupe(queries)[:8]



def _infer_evidence_intent(text: str, *, numbers: list[str], dates: list[str]) -> str:
    compact = " ".join(text.split())
    if any(hint in compact for hint in COMPARE_HINTS):
        return "comparison"
    if numbers or dates:
        return "number"
    return "fact"



def _pick_group_intent(targets: list[RetrievalTarget]) -> str:
    intents = [target.evidence_intent for target in targets]
    if "comparison" in intents:
        return "comparison"
    if "number" in intents:
        return "number"
    return intents[0] if intents else "fact"



def _memory_anchor_text(prior_memories: list[dict]) -> str:
    if not prior_memories:
        return ""
    summary = " ".join(str(prior_memories[-1].get("summary", "")).split())
    return summary[:80]



def _estimate_evidence_density(texts: list[str]) -> float:
    if not texts:
        return 0.0
    score = 0.0
    for text in texts:
        if any(hint in text for hint in SPECIFIC_EVIDENCE_HINTS):
            score += 0.40
        if any(char.isdigit() for char in text) and any(hint in text for hint in SPECIFIC_EVIDENCE_HINTS):
            score += 0.35
        if any(hint in text for hint in CLAUSE_CONSEQUENCE_HINTS):
            score += 0.35
    return min(1.0, score / max(1, len(texts)))



def _is_generic_context_only(texts: list[str]) -> bool:
    if not texts:
        return True
    for text in texts:
        if any(hint in text for hint in SPECIFIC_EVIDENCE_HINTS):
            return False
        if any(hint in text for hint in CLAUSE_CONSEQUENCE_HINTS):
            return False
    if not any(any(hint in text for hint in GENERIC_CONTEXT_HINTS) for text in texts):
        return False
    return True



def _looks_like_clause_consequence_question(text: str) -> bool:
    compact = " ".join((text or "").split())
    return any(hint in compact for hint in ("处罚", "扣减", "减分", "期限", "是否会被扣减", "行政处罚"))



def _has_clause_consequence(texts: list[str]) -> bool:
    return any(any(hint in text for hint in CLAUSE_CONSEQUENCE_HINTS) for text in texts)



def _dedupe(items) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join(str(item).split())
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
