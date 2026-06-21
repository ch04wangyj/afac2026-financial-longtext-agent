"""Utilities for deterministic retrieval-system probes."""

from __future__ import annotations

import re

from agent.schemas import RetrievalResult


def find_doc_rank(results: list[RetrievalResult], target_doc_id: str) -> int | None:
    """Return 1-based rank of the first result from target_doc_id."""
    for rank, result in enumerate(results, start=1):
        if result.doc_id == target_doc_id:
            return rank
    return None


def find_answer_chunk_rank(
    results: list[RetrievalResult],
    *,
    target_doc_id: str,
    answer_terms: tuple[str, ...],
    answer_match_mode: str = "all_terms_in_one_chunk",
) -> int | None:
    """Return 1-based rank of first matching target-doc chunk under the requested answer match mode."""
    normalized_terms = tuple(_normalize_match_text(term) for term in answer_terms)
    for rank, result in enumerate(results, start=1):
        if result.doc_id != target_doc_id:
            continue
        text = _normalize_match_text(result.evidence_text)
        if answer_match_mode == "any_term_in_target_doc_chunk":
            if any(term in text for term in normalized_terms):
                return rank
            continue
        if all(term in text for term in normalized_terms):
            return rank
    return None


def find_answer_coverage_rank(
    results: list[RetrievalResult],
    *,
    target_doc_id: str,
    answer_terms: tuple[str, ...],
) -> int | None:
    """Return the smallest rank prefix whose target-doc chunks jointly cover all answer terms."""
    normalized_terms = tuple(_normalize_match_text(term) for term in answer_terms)
    covered: set[str] = set()
    for rank, result in enumerate(results, start=1):
        if result.doc_id != target_doc_id:
            continue
        text = _normalize_match_text(result.evidence_text)
        for term in normalized_terms:
            if term in text:
                covered.add(term)
        if len(covered) == len(normalized_terms):
            return rank
    return None


def summarize_probe_results(
    *,
    method: str,
    results: list[RetrievalResult],
    target_doc_id: str,
    answer_terms: tuple[str, ...],
    answer_match_mode: str = "all_terms_in_one_chunk",
    top_k: int = 10,
) -> dict:
    """Summarize retrieval results for ranking diagnostics using answer_chunk_rank as the primary default signal."""
    doc_rank = find_doc_rank(results, target_doc_id)
    answer_rank = find_answer_chunk_rank(
        results,
        target_doc_id=target_doc_id,
        answer_terms=answer_terms,
        answer_match_mode=answer_match_mode,
    )
    return {
        "method": method,
        "doc_rank": doc_rank,
        "answer_chunk_rank": answer_rank,
        "target_doc_in_top_5": doc_rank is not None and doc_rank <= 5,
        "target_doc_in_top_10": doc_rank is not None and doc_rank <= 10,
        "answer_chunk_in_top_5": answer_rank is not None and answer_rank <= 5,
        "answer_chunk_in_top_10": answer_rank is not None and answer_rank <= 10,
        "top_results": [_result_summary(result, rank) for rank, result in enumerate(results[:top_k], start=1)],
    }


def _normalize_match_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"\s+", "", value)
    value = value.replace("，", ",")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    return value


def _result_summary(result: RetrievalResult, rank: int) -> dict:
    text = " ".join(result.evidence_text.split())
    return {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "score": round(float(result.score), 6),
        "source": result.source,
        "query": result.query,
        "page": result.metadata.get("page"),
        "section": result.metadata.get("section"),
        "title": result.metadata.get("title"),
        "snippet": text[:220] + ("..." if len(text) > 220 else ""),
        "score_breakdown": result.metadata.get("score_breakdown"),
        "rerank_features": result.metadata.get("rerank_features"),
    }
