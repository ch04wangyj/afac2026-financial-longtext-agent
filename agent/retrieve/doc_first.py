"""Document-first sparse retrieval helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import Chunk, RetrievalResult

CORPORATE_SUFFIXES = (
    "集团股份有限公司",
    "股份有限公司",
    "集团有限公司",
    "有限公司",
    "集团",
    "公司",
)

FINANCIAL_RATIO_DISTRACTOR_HINTS = (
    "现金分红(含税)",
    "现金分红金额",
    "现金分红/归属于母公司所有者的净利润",
    "现金分红/归属于母公司股东的净利润",
    "归属于母公司所有者权益",
)

NARRATIVE_ACTION_HINTS = (
    "持续推出实施",
    "连续四年",
    "回购计划",
    "股份回购",
    "回购注销",
    "股权激励计划",
    "维护公司市值稳定",
)


@dataclass(frozen=True)
class DocumentCandidate:
    doc_id: str
    score: float
    matched_terms: tuple[str, ...]


def normalize_company_like_text(text: str) -> str:
    value = "".join(str(text or "").split())
    for suffix in CORPORATE_SUFFIXES:
        if len(value) > len(suffix) and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def rank_document_candidates(
    chunks: list[Chunk],
    *,
    keyword_bundles: list[tuple[str, ...]],
    top_n: int = 8,
) -> list[DocumentCandidate]:
    doc_text: dict[str, list[str]] = defaultdict(list)
    doc_title: dict[str, str] = {}
    for chunk in chunks:
        doc_title.setdefault(chunk.doc_id, str(chunk.metadata.get("title", "")))
        doc_text[chunk.doc_id].append(" ".join([chunk.section, chunk.clause_id, chunk.text[:2000]]))

    scores: list[DocumentCandidate] = []
    for doc_id, parts in doc_text.items():
        title = doc_title.get(doc_id, "")
        haystack = " ".join([title, *parts])
        normalized_haystack = normalize_company_like_text(haystack)
        score = 0.0
        matched: list[str] = []
        for bundle in keyword_bundles:
            for raw_term in bundle:
                term = str(raw_term).strip()
                if not term:
                    continue
                normalized_term = normalize_company_like_text(term)
                if term in haystack or (normalized_term and normalized_term in normalized_haystack):
                    matched.append(term)
                    score += _term_weight(term)
        if score > 0:
            scores.append(DocumentCandidate(doc_id=doc_id, score=score, matched_terms=tuple(sorted(set(matched)))))

    return sorted(scores, key=lambda item: item.score, reverse=True)[:top_n]


def retrieve_candidate_doc_ids(
    doc_index: DocumentSearchIndex,
    *,
    keyword_bundles: list[tuple[str, ...]],
    top_docs: int = 8,
) -> list[str]:
    ranked_lists: list[list[str]] = []
    for bundle in keyword_bundles:
        query = " ".join(bundle)
        ranked_lists.append(doc_index.search_doc_ids(query, top_k=top_docs))

    scores: dict[str, float] = {}
    for doc_ids in ranked_lists:
        for rank, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_docs]]


def aggregate_chunk_hits_to_doc_ids(
    index: BM25SearchIndex,
    *,
    keyword_bundles: list[tuple[str, ...]],
    top_docs: int = 12,
    per_query_top_k: int = 80,
    scoring_mode: str | None = "bm25f_lite",
) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for bundle in keyword_bundles:
        query = " ".join(bundle)
        results = index.search(query, top_k=per_query_top_k, source="doc_agg", scoring_mode=scoring_mode)
        for rank, result in enumerate(results, start=1):
            scores[result.doc_id] += 1.0 / (60 + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_docs]]


def expand_top_doc_structural_neighbors(
    seed_results: list[RetrievalResult],
    *,
    candidate_neighbors: list[RetrievalResult],
    top_docs: set[str],
) -> list[RetrievalResult]:
    """Add same-doc structural neighbors for top candidate docs."""
    existing = {item.chunk_id for item in seed_results}
    expanded = list(seed_results)
    strong_seed_docs = {item.doc_id for item in seed_results if item.doc_id in top_docs}
    for neighbor in candidate_neighbors:
        if neighbor.doc_id not in strong_seed_docs:
            continue
        if neighbor.chunk_id in existing:
            continue
        expanded.append(neighbor)
        existing.add(neighbor.chunk_id)
    return expanded


def retrieve_doc_first(
    index: BM25SearchIndex,
    *,
    keyword_bundles: list[tuple[str, ...]],
    top_docs: int = 12,
    top_k: int = 30,
) -> list[RetrievalResult]:
    candidate_doc_ids = aggregate_chunk_hits_to_doc_ids(
        index,
        keyword_bundles=keyword_bundles,
        top_docs=top_docs,
        per_query_top_k=max(80, top_k * 2),
        scoring_mode="bm25f_lite",
    )
    filter_doc_ids = set(candidate_doc_ids)
    ranked_lists = []
    for bundle in keyword_bundles:
        query = " ".join(bundle)
        ranked_lists.append(
            index.search(query, top_k=top_k, filter_doc_ids=filter_doc_ids, source="doc_first:broad_bm25")
        )
        ranked_lists.append(
            index.search(
                query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
                source="doc_first:bm25f_lite",
                scoring_mode="bm25f_lite",
            )
        )
    fused = reciprocal_rank_fusion(ranked_lists, top_k=top_k * 2)

    top_doc_set = set(candidate_doc_ids[: max(3, min(5, len(candidate_doc_ids)))])
    neighbor_pool: list[RetrievalResult] = []
    seeded_pages: set[tuple[str, int]] = set()
    seeded_sections: set[tuple[str, str]] = set()
    for item in fused:
        if item.doc_id not in top_doc_set:
            continue
        page = item.metadata.get("page")
        if page is not None:
            seeded_pages.add((item.doc_id, int(page)))
        section = str(item.metadata.get("section") or "").strip()
        if section:
            seeded_sections.add((item.doc_id, section))
    for doc_id in top_doc_set:
        seen_neighbor_ids: set[str] = set()
        for page_doc_id, page in seeded_pages:
            if page_doc_id != doc_id:
                continue
            for chunk in index.get_same_page_chunks(doc_id, page, allowed_doc_ids=top_doc_set):
                if chunk.chunk_id in seen_neighbor_ids:
                    continue
                neighbor_pool.append(index.result_from_chunk(chunk, score=0.0, source="doc_first:neighbor", query=""))
                seen_neighbor_ids.add(chunk.chunk_id)
        for section_doc_id, section in seeded_sections:
            if section_doc_id != doc_id:
                continue
            for chunk in index.get_same_section_chunks(doc_id, section, allowed_doc_ids=top_doc_set):
                if chunk.chunk_id in seen_neighbor_ids:
                    continue
                neighbor_pool.append(index.result_from_chunk(chunk, score=0.0, source="doc_first:neighbor", query=""))
                seen_neighbor_ids.add(chunk.chunk_id)
    fused = expand_top_doc_structural_neighbors(fused, candidate_neighbors=neighbor_pool, top_docs=top_doc_set)

    doc_scores = {doc_id: float(len(candidate_doc_ids) - rank) for rank, doc_id in enumerate(candidate_doc_ids, start=0)}
    for result in fused:
        result.metadata["doc_first_score"] = doc_scores.get(result.doc_id, 0.0)
        result.metadata["doc_first_candidates"] = list(candidate_doc_ids)
    return rerank_doc_first_chunks(fused, keyword_bundles=keyword_bundles, doc_scores=doc_scores)[:top_k]


def rerank_doc_first_chunks(
    results: list[RetrievalResult],
    *,
    keyword_bundles: list[tuple[str, ...]],
    doc_scores: dict[str, float],
) -> list[RetrievalResult]:
    rescored = []
    max_doc_score = max(doc_scores.values(), default=1.0) or 1.0
    max_retrieval_score = max((float(item.score) for item in results), default=1.0) or 1.0
    bundle_terms = {str(term).strip() for bundle in keyword_bundles for term in bundle if str(term).strip()}
    query_year_terms = {term for term in bundle_terms if term.isdigit() and len(term) == 4}
    section_support: dict[tuple[str, str], float] = defaultdict(float)
    section_hit_counts: dict[tuple[str, str], int] = defaultdict(int)
    page_support: dict[tuple[str, int], float] = defaultdict(float)
    page_hit_counts: dict[tuple[str, int], int] = defaultdict(int)
    matched_terms_by_chunk: dict[str, list[str]] = {}
    raw_match_quality_by_chunk: dict[str, float] = {}
    for item in results:
        section_key = str(item.metadata.get("section") or "").strip().lower()
        page = item.metadata.get("page")
        if not section_key:
            section_key = ""
        text = item.evidence_text
        matched_terms = [term for term in bundle_terms if term in text]
        matched_terms_by_chunk[item.chunk_id] = matched_terms
        raw_match_quality_by_chunk[item.chunk_id] = sum(_term_weight(term) * max(1, len(term)) for term in matched_terms)
        lexical_hits = len(matched_terms)
        if lexical_hits <= 0:
            continue
        if section_key:
            section_support[(item.doc_id, section_key)] += min(2.0, lexical_hits) * max(0.0, float(item.score))
            section_hit_counts[(item.doc_id, section_key)] += 1
        if page is not None:
            page_key = (item.doc_id, int(page))
            page_support[page_key] += min(2.0, lexical_hits) * max(0.0, float(item.score))
            page_hit_counts[page_key] += 1
    max_match_quality = max(raw_match_quality_by_chunk.values(), default=0.0) or 1.0
    for result in results:
        text = result.evidence_text
        number_density_bonus = 0.05 if any(ch.isdigit() for ch in text) else 0.0
        table_bonus = 0.15 if result.metadata.get("chunk_type") == "table" or result.metadata.get("section") == "table" else 0.0
        statement_bonus = 0.0
        narrative_action_bonus = 0.18 if any(hint in text for hint in NARRATIVE_ACTION_HINTS) else 0.0
        distractor_penalty = 0.0
        ratio_distractor_penalty = 0.28 if any(hint in text for hint in FINANCIAL_RATIO_DISTRACTOR_HINTS) else 0.0
        doc_bonus = (doc_scores.get(result.doc_id, 0.0) / max_doc_score) * 1.45
        doc_keyword_cooccurrence_bonus = 0.0
        matched_terms = matched_terms_by_chunk.get(result.chunk_id, [term for term in bundle_terms if term in text])
        term_hits = len(matched_terms)
        if term_hits >= 2 and any(hint in text for hint in NARRATIVE_ACTION_HINTS):
            doc_keyword_cooccurrence_bonus = 0.12
        lexical_concentration_bonus = 0.0
        if term_hits >= 2:
            lexical_concentration_bonus = min(0.24, 0.06 * term_hits)
        term_specificity_bonus = 0.0
        if matched_terms:
            term_specificity_bonus = min(0.30, sum(min(0.12, 0.015 * len(term)) for term in matched_terms))
        section_density_bonus = 0.0
        section_key = str(result.metadata.get("section") or "").strip().lower()
        if section_key:
            raw_section_support = section_support.get((result.doc_id, section_key), 0.0)
            if raw_section_support > 0:
                section_density_bonus = min(0.35, 0.06 * raw_section_support)
        section_peer_support_bonus = 0.0
        if section_key and term_hits >= 2:
            peer_hits = section_hit_counts.get((result.doc_id, section_key), 0)
            if peer_hits >= 2:
                section_peer_support_bonus = 0.10
        page_peer_support_bonus = 0.0
        page_density_bonus = 0.0
        page = result.metadata.get("page")
        if page is not None:
            page_key = (result.doc_id, int(page))
            peer_page_hits = page_hit_counts.get(page_key, 0)
            raw_page_support = page_support.get(page_key, 0.0)
            if raw_page_support > 0 and peer_page_hits >= 2:
                page_density_bonus = min(0.42, 0.08 * raw_page_support)
            if term_hits >= 2 and peer_page_hits >= 2:
                page_peer_support_bonus = 0.16
        year_consistency_bonus = 0.0
        if query_year_terms and any(year in text for year in query_year_terms):
            year_consistency_bonus = 0.16
        query_match_quality = raw_match_quality_by_chunk.get(result.chunk_id, 0.0) / max_match_quality
        base = (float(result.score) / max_retrieval_score) * query_match_quality
        rerank_score = (
            base
            + doc_bonus
            + doc_keyword_cooccurrence_bonus
            + lexical_concentration_bonus
            + term_specificity_bonus
            + section_density_bonus
            + section_peer_support_bonus
            + page_density_bonus
            + page_peer_support_bonus
            + year_consistency_bonus
            + narrative_action_bonus
            + number_density_bonus
            + table_bonus
            + statement_bonus
            - distractor_penalty
            - ratio_distractor_penalty
        )
        result.metadata["doc_first_rerank_features"] = {
            "base": round(base, 6),
            "query_match_quality": round(query_match_quality, 6),
            "doc_bonus": round(doc_bonus, 6),
            "doc_keyword_cooccurrence_bonus": doc_keyword_cooccurrence_bonus,
            "lexical_concentration_bonus": round(lexical_concentration_bonus, 6),
            "term_specificity_bonus": round(term_specificity_bonus, 6),
            "section_density_bonus": round(section_density_bonus, 6),
            "section_peer_support_bonus": round(section_peer_support_bonus, 6),
            "page_density_bonus": round(page_density_bonus, 6),
            "page_peer_support_bonus": round(page_peer_support_bonus, 6),
            "year_consistency_bonus": year_consistency_bonus,
            "narrative_action_bonus": narrative_action_bonus,
            "number_density_bonus": number_density_bonus,
            "table_bonus": table_bonus,
            "statement_bonus": statement_bonus,
            "distractor_penalty": distractor_penalty,
            "ratio_distractor_penalty": ratio_distractor_penalty,
        }
        result.score = float(rerank_score)
        result.source = "doc_first_chunk_rerank"
        rescored.append(result)
    return sorted(rescored, key=lambda item: item.score, reverse=True)


def _term_weight(term: str) -> float:
    if any(ch.isdigit() for ch in term):
        return 1.2
    if "净利润" in term or "利润" in term:
        return 1.1
    if len(term) >= 4:
        return 1.4
    return 1.0
