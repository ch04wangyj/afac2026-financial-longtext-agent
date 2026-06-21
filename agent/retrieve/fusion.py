"""排序融合工具。"""

from __future__ import annotations

from agent.schemas import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 30,
    doc_rescue_top_n: int = 5,
    weights: list[float] | None = None,
) -> list[RetrievalResult]:
    """RRF 融合 chunk，并保留文档级强候选的代表 chunk。

    多 query 可能在同一文档命中不同 chunk。纯 chunk-level RRF 会把这种文档
    误判为“各路都不稳定”，因此额外聚合每路首次出现的文档排名，并在最终
    Top-K 尾部为文档级 Top-N 补一个代表 chunk。
    """
    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}
    doc_scores: dict[str, float] = {}
    doc_best: dict[str, RetrievalResult] = {}
    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")
    effective_weights = weights or [1.0] * len(ranked_lists)
    for results, list_weight in zip(ranked_lists, effective_weights):
        seen_docs: set[str] = set()
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + float(list_weight) / (k + rank)
            if result.chunk_id not in best or result.score > best[result.chunk_id].score:
                best[result.chunk_id] = result
            if result.doc_id not in seen_docs:
                doc_scores[result.doc_id] = doc_scores.get(result.doc_id, 0.0) + float(list_weight) / (k + rank)
                seen_docs.add(result.doc_id)
            if result.doc_id not in doc_best or result.score > doc_best[result.doc_id].score:
                doc_best[result.doc_id] = result

    fused: list[RetrievalResult] = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        # 保留该 chunk 在各路检索中的最佳原始结果，同时把融合分写回 score。
        result = best[chunk_id]
        result.score = float(score)
        result.source = f"rrf:{result.source}"
        fused.append(result)
    if doc_rescue_top_n > 0 and top_k > 0:
        _rescue_top_documents(
            fused,
            top_docs=[doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)],
            doc_best=doc_best,
            doc_scores=doc_scores,
            top_n=doc_rescue_top_n,
            top_k=top_k,
        )
    return fused


def _rescue_top_documents(
    fused: list[RetrievalResult],
    *,
    top_docs: list[str],
    doc_best: dict[str, RetrievalResult],
    doc_scores: dict[str, float],
    top_n: int,
    top_k: int,
) -> None:
    protected_docs = set(top_docs[:top_n])
    represented = {item.doc_id for item in fused}
    for doc_id in top_docs[:top_n]:
        if doc_id in represented or doc_id not in doc_best:
            continue
        rescue = doc_best[doc_id]
        rescue.score = float(doc_scores.get(doc_id, 0.0))
        rescue.source = f"rrf_doc_rescue:{rescue.source}"
        if len(fused) >= top_k:
            drop_index = next(
                (index for index in range(len(fused) - 1, -1, -1) if fused[index].doc_id not in protected_docs),
                len(fused) - 1,
            )
            dropped = fused.pop(drop_index)
            represented.discard(dropped.doc_id)
        fused.append(rescue)
        represented.add(doc_id)
