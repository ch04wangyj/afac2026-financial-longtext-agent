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



def _dedupe(items) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join(str(item).split())
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
