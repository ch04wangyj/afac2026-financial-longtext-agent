"""排序融合工具。"""

from __future__ import annotations

from agent.schemas import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 30,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion：奖励多路检索共同命中的 chunk。"""
    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            if result.chunk_id not in best or result.score > best[result.chunk_id].score:
                best[result.chunk_id] = result

    fused: list[RetrievalResult] = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        # 保留该 chunk 在各路检索中的最佳原始结果，同时把融合分写回 score。
        result = best[chunk_id]
        result.score = float(score)
        result.source = f"rrf:{result.source}"
        fused.append(result)
    return fused
