"""Lexical and structural reranking for sparse retrieval candidates."""

from __future__ import annotations

from agent.index.tokenizer import tokenize
from agent.retrieve.targets import RetrievalTarget, question_with_options
from agent.schemas import Question, RetrievalResult



def rerank_retrieval_results(
    question: Question,
    target: RetrievalTarget,
    results: list[RetrievalResult],
    *,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    if not results:
        return []
    max_score = max(float(item.score) for item in results) or 1.0
    query_terms = set(tokenize(question_with_options(question), mode="mixed"))
    must_terms = target.must_terms[:]
    option_terms = target.option_terms[:]
    seen_texts: dict[str, int] = {}
    rescored: list[RetrievalResult] = []

    for result in results:
        text = result.evidence_text
        evidence_terms = set(tokenize(text, mode="mixed"))
        normalized_bm25 = float(result.score) / max_score
        must_hits = sum(1 for term in must_terms if term and term in text)
        option_hits = sum(1 for term in option_terms if term and term[:16] in text)
        number_overlap = len(set(target.numbers) & set(result.metadata.get("numbers", [])))
        date_overlap = len(set(target.dates) & set(result.metadata.get("dates", [])))
        query_overlap = len(query_terms & evidence_terms) / max(1, len(query_terms))
        structure_bonus = 0.0
        if result.metadata.get("clause_id"):
            structure_bonus += 0.15
        if result.metadata.get("section") and str(result.metadata.get("section")) not in {"", "document"}:
            structure_bonus += 0.05
        if result.metadata.get("chunk_type") in {"table", "figure"}:
            structure_bonus += 0.10 if target.evidence_intent in {"number", "comparison"} else 0.04
        doc_bonus = 0.10 if target.doc_scope and result.doc_id in set(target.doc_scope) else 0.0
        duplicate_penalty = 0.08 if seen_texts.get(text.strip(), 0) else 0.0
        rerank_score = (
            normalized_bm25
            + must_hits * 0.18
            + option_hits * 0.08
            + number_overlap * 0.12
            + date_overlap * 0.12
            + query_overlap * 0.35
            + structure_bonus
            + doc_bonus
            - duplicate_penalty
        )
        result.metadata["rerank_features"] = {
            "normalized_bm25": round(normalized_bm25, 6),
            "must_hits": must_hits,
            "option_hits": option_hits,
            "number_overlap": number_overlap,
            "date_overlap": date_overlap,
            "query_overlap": round(query_overlap, 6),
            "structure_bonus": round(structure_bonus, 6),
            "doc_bonus": round(doc_bonus, 6),
            "duplicate_penalty": round(duplicate_penalty, 6),
        }
        result.metadata["rerank_score"] = float(rerank_score)
        result.score = float(rerank_score)
        rescored.append(result)
        seen_texts[text.strip()] = seen_texts.get(text.strip(), 0) + 1

    ranked = sorted(rescored, key=lambda item: item.score, reverse=True)
    return ranked[:top_k] if top_k else ranked
