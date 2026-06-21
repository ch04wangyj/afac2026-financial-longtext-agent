"""Deterministic context expansion and evidence-pack helpers."""

from __future__ import annotations

from agent.schemas import EvidencePack, Question, RetrievalResult



def build_evidence_packs(
    index,
    question: Question,
    results: list[RetrievalResult],
    *,
    max_packs: int = 6,
    neighbor_window: int = 1,
    max_chunks_per_pack: int = 4,
    max_chars_per_pack: int = 2200,
) -> list[EvidencePack]:
    if not results:
        return []
    allowed_doc_ids = set(question.doc_ids) if question.doc_ids else {item.doc_id for item in results}
    anchors = _select_anchor_results(question, results, max_packs=max_packs)
    packs: list[EvidencePack] = []
    for rank, anchor in enumerate(anchors):
        if anchor.doc_id not in allowed_doc_ids:
            continue
        chunk = index.get_chunk(anchor.chunk_id)
        if chunk is None:
            continue
        members = _expand_anchor(
            index,
            anchor,
            allowed_doc_ids=allowed_doc_ids,
            neighbor_window=neighbor_window,
            max_chunks_per_pack=max_chunks_per_pack,
            max_chars_per_pack=max_chars_per_pack,
        )
        if not members:
            continue
        pack_id = f"pack::{anchor.chunk_id}"
        pack_text = "\n\n".join(item.text for item in members)
        packs.append(
            EvidencePack(
                pack_id=pack_id,
                doc_id=anchor.doc_id,
                anchor_chunk_id=anchor.chunk_id,
                member_chunk_ids=[item.chunk_id for item in members],
                score=float(anchor.score),
                source=anchor.source,
                query=anchor.query,
                text=pack_text,
                metadata={
                    "pages": sorted({item.page for item in members if item.page is not None}),
                    "section": chunk.section,
                    "clause_id": chunk.clause_id,
                    "chunk_types": [item.metadata.get("chunk_type", "text") for item in members],
                    "expansion_reason": _expansion_reason(chunk),
                    "anchor_rank": rank,
                    "anchor_score": float(anchor.score),
                },
            )
        )
    return _merge_overlapping_packs(packs, max_chars_per_pack=max_chars_per_pack)



def select_results_from_packs(
    index,
    question: Question,
    packs: list[EvidencePack],
    *,
    top_k: int = 8,
    max_chars: int = 6000,
) -> list[RetrievalResult]:
    if not packs:
        return []
    selected: list[RetrievalResult] = []
    selected_chunk_ids: set[str] = set()
    used_chars = 0
    ranked_packs = sorted(packs, key=lambda item: item.score, reverse=True)

    doc_quota = list(dict.fromkeys(question.doc_ids or []))
    if len(doc_quota) > 1:
        best_by_doc: dict[str, EvidencePack] = {}
        for pack in ranked_packs:
            best_by_doc.setdefault(pack.doc_id, pack)
        for doc_id in doc_quota:
            pack = best_by_doc.get(doc_id)
            if pack is None:
                continue
            used_chars = _select_pack_results(
                index,
                pack,
                selected,
                selected_chunk_ids,
                used_chars,
                max_chars=max_chars,
                top_k=top_k,
            )
            if len(selected) >= top_k:
                return selected[:top_k]

    for pack in ranked_packs:
        used_chars = _select_pack_results(
            index,
            pack,
            selected,
            selected_chunk_ids,
            used_chars,
            max_chars=max_chars,
            top_k=top_k,
        )
        if len(selected) >= top_k:
            break
    return selected[:top_k]



def _select_anchor_results(question: Question, results: list[RetrievalResult], *, max_packs: int) -> list[RetrievalResult]:
    anchors: list[RetrievalResult] = []
    seen_chunks: set[str] = set()
    seen_docs: set[str] = set()
    for doc_id in dict.fromkeys(question.doc_ids or []):
        for result in results:
            if result.doc_id == doc_id and result.chunk_id not in seen_chunks:
                anchors.append(result)
                seen_chunks.add(result.chunk_id)
                seen_docs.add(doc_id)
                break
    for result in results:
        if result.chunk_id in seen_chunks:
            continue
        anchors.append(result)
        seen_chunks.add(result.chunk_id)
        seen_docs.add(result.doc_id)
        if len(anchors) >= max_packs:
            break
    return anchors[:max_packs]



def _expand_anchor(
    index,
    anchor: RetrievalResult,
    *,
    allowed_doc_ids: set[str],
    neighbor_window: int,
    max_chunks_per_pack: int,
    max_chars_per_pack: int,
):
    chunk = index.get_chunk(anchor.chunk_id)
    if chunk is None:
        return []
    candidates = []
    seen_ids: set[str] = set()

    def add(items):
        for item in items:
            if item.doc_id not in allowed_doc_ids or item.chunk_id in seen_ids:
                continue
            candidates.append(item)
            seen_ids.add(item.chunk_id)

    add([chunk])
    if chunk.clause_id:
        add(index.get_same_clause_chunks(chunk.doc_id, chunk.clause_id, allowed_doc_ids=allowed_doc_ids))
    elif chunk.section:
        add(index.get_same_section_chunks(chunk.doc_id, chunk.section, allowed_doc_ids=allowed_doc_ids))
    if chunk.metadata.get("chunk_type") in {"table", "figure"}:
        add(index.get_same_page_chunks(chunk.doc_id, chunk.page, allowed_doc_ids=allowed_doc_ids))
    add(index.get_doc_neighbors(anchor.chunk_id, left=neighbor_window, right=neighbor_window, allowed_doc_ids=allowed_doc_ids))

    selected = []
    used_chars = 0
    for item in candidates:
        if len(selected) >= max_chunks_per_pack:
            break
        text = item.text.strip()
        if not text:
            continue
        if used_chars + len(text) > max_chars_per_pack and selected:
            continue
        selected.append(item)
        used_chars += len(text)
    return selected



def _merge_overlapping_packs(packs: list[EvidencePack], *, max_chars_per_pack: int) -> list[EvidencePack]:
    merged: list[EvidencePack] = []
    for pack in sorted(packs, key=lambda item: item.score, reverse=True):
        overlap = next((existing for existing in merged if set(existing.member_chunk_ids) & set(pack.member_chunk_ids)), None)
        if overlap is None:
            merged.append(pack)
            continue
        merged_chunk_ids = []
        seen: set[str] = set()
        for chunk_id in [*overlap.member_chunk_ids, *pack.member_chunk_ids]:
            if chunk_id not in seen:
                merged_chunk_ids.append(chunk_id)
                seen.add(chunk_id)
        overlap.member_chunk_ids = merged_chunk_ids
        overlap.score = max(overlap.score, pack.score)
        overlap.metadata["expansion_reason"] = "merged_overlap"
        overlap.text = overlap.text if len(overlap.text) >= len(pack.text) else pack.text[:max_chars_per_pack]
    return merged



def _select_pack_results(index, pack: EvidencePack, selected, selected_chunk_ids: set[str], used_chars: int, *, max_chars: int, top_k: int) -> int:
    for member_rank, chunk_id in enumerate(pack.member_chunk_ids):
        if len(selected) >= top_k or chunk_id in selected_chunk_ids:
            continue
        chunk = index.get_chunk(chunk_id)
        if chunk is None:
            continue
        text = chunk.text.strip()
        if not text:
            continue
        if used_chars + len(text) > max_chars and selected:
            continue
        result = index.result_from_chunk(
            chunk,
            score=float(pack.score) - member_rank * 1e-4,
            source=f"{pack.source}:pack",
            query=pack.query,
        )
        result.metadata["pack_id"] = pack.pack_id
        result.metadata["pack_anchor_chunk_id"] = pack.anchor_chunk_id
        result.metadata["pack_role"] = "anchor" if chunk_id == pack.anchor_chunk_id else "context"
        result.metadata["expansion_reason"] = pack.metadata.get("expansion_reason", "neighbor_window")
        selected.append(result)
        selected_chunk_ids.add(chunk_id)
        used_chars += len(text)
    return used_chars



def _expansion_reason(chunk) -> str:
    if chunk.clause_id:
        return "same_clause"
    if chunk.metadata.get("chunk_type") in {"table", "figure"}:
        return "same_page_structure"
    if chunk.section:
        return "same_section"
    return "neighbor_window"
