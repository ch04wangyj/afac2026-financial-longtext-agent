"""Conservative sparse expansion helpers for LogicRAG retrieval."""

from __future__ import annotations

from agent.index.tokenizer import tokenize
from agent.retrieve.structured_queries import extract_query_entities
from agent.retrieve.targets import RetrievalTarget
from agent.schemas import Question, RetrievalResult



def build_short_hypothetical_query(target: RetrievalTarget, max_terms: int = 4) -> str:
    terms = _dedupe([*target.must_terms, *target.option_terms, *target.numbers, *target.dates])
    return " ".join(terms[:max_terms])



def build_sparse_feedback_query(
    question: Question,
    target: RetrievalTarget,
    seed_results: list[RetrievalResult],
    *,
    idf_lookup: dict[str, float] | None = None,
    max_terms: int = 4,
) -> str:
    base_query = " ".join(target.query_variants[:2]) or question.question
    existing_terms = set(tokenize(base_query, mode="mixed"))
    candidate_terms: list[tuple[float, str]] = []

    def consider(term: str, base_bonus: float = 0.0) -> None:
        term = " ".join(str(term).split())
        if len(term) < 2:
            return
        term_tokens = tokenize(term, mode="mixed")
        if not term_tokens:
            return
        if all(token in existing_terms for token in term_tokens):
            return
        rarity = max((idf_lookup or {}).get(token, 0.0) for token in term_tokens)
        candidate_terms.append((rarity + base_bonus, term))

    for result in seed_results[:6]:
        title = str(result.metadata.get("title", ""))
        clause_id = str(result.metadata.get("clause_id", ""))
        section = str(result.metadata.get("section", ""))
        chunk_type = str(result.metadata.get("chunk_type", ""))
        if title:
            consider(title, 0.6)
        if clause_id:
            consider(clause_id, 1.0)
        if section and section not in {"table", "figure", "document"}:
            consider(section, 0.4)
        if chunk_type in {"table", "figure"}:
            consider(chunk_type, 0.2)
        for term in extract_query_entities(result.evidence_text)[:8]:
            consider(term, 0.2)
        for term in result.metadata.get("numbers", [])[:4]:
            consider(term, 0.15)
        for term in result.metadata.get("dates", [])[:4]:
            consider(term, 0.15)

    selected_terms = [term for _score, term in sorted(candidate_terms, key=lambda item: item[0], reverse=True)]
    selected_terms = _dedupe(selected_terms)[:max_terms]
    if not selected_terms:
        return ""
    return " ".join([base_query, *selected_terms]).strip()



def _dedupe(items) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join(str(item).split())
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
