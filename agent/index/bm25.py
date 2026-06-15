"""轻量 BM25 检索实现。

这里保留纯 Python fallback，避免 bm25s/rank_bm25 安装问题阻塞比赛链路。
索引只使用词法特征，不使用 embedding。
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from pathlib import Path

from agent.index.tokenizer import tokenize, tokenize_chunk
from agent.schemas import Chunk, RetrievalResult


class SimpleBM25:
    """倒排表加速版 BM25，用于 chunk 级稀疏检索。"""

    def __init__(
        self,
        corpus_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        max_query_terms: int = 160,
    ) -> None:
        # 预计算文档长度、词频、倒排表和 IDF，后续查询只遍历命中的 term posting。
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.max_query_terms = max_query_terms
        self.doc_count = len(corpus_tokens)
        self.doc_lens = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, self.doc_count)
        self.term_freqs = [Counter(tokens) for tokens in corpus_tokens]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: Counter[str] = Counter()
        for doc_idx, freqs in enumerate(self.term_freqs):
            for term, tf in freqs.items():
                self.postings.setdefault(term, []).append((doc_idx, tf))
            doc_freq.update(freqs.keys())
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        """计算单个文档对查询的 BM25 分数，主要用于测试和调试。"""
        score = 0.0
        freqs = self.term_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx] or 1
        for token in query_tokens:
            tf = freqs.get(token, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1e-9, self.avgdl))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        """返回全量文档分数；语义直观但比倒排搜索慢。"""
        return [self.score(query_tokens, idx) for idx in range(self.doc_count)]

    def get_score_items(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        """只遍历查询词命中的 posting，返回非零分文档。"""
        scores: dict[int, float] = {}
        for token in self._select_query_terms(query_tokens):
            idf = self.idf.get(token, 0.0)
            if idf <= 0:
                continue
            for doc_idx, tf in self.postings.get(token, []):
                doc_len = self.doc_lens[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1e-9, self.avgdl))
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        return list(scores.items())

    def _select_query_terms(self, query_tokens: list[str]) -> list[str]:
        """长查询只保留高 IDF 词，防止选项拼接后查询过宽。"""
        unique = set(query_tokens)
        if len(unique) <= self.max_query_terms:
            return list(unique)
        return sorted(unique, key=lambda token: self.idf.get(token, 0.0), reverse=True)[: self.max_query_terms]


class BM25SearchIndex:
    """可持久化的 chunk 检索索引。"""

    def __init__(
        self,
        chunks: list[Chunk],
        corpus_tokens: list[list[str]] | None = None,
        tokenizer_mode: str = "mixed",
    ) -> None:
        self.chunks = chunks
        self.tokenizer_mode = tokenizer_mode
        self.corpus_tokens = corpus_tokens or [tokenize_chunk(chunk, mode=tokenizer_mode) for chunk in chunks]
        self.engine = SimpleBM25(self.corpus_tokens)

    @classmethod
    def build(cls, chunks: list[Chunk], tokenizer_mode: str = "mixed") -> "BM25SearchIndex":
        """从 chunk 列表构建索引。"""
        return cls(chunks, tokenizer_mode=tokenizer_mode)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
        source: str = "bm25",
    ) -> list[RetrievalResult]:
        """执行 BM25 检索，并可按 doc_id 限定候选文档。"""
        query_tokens = tokenize(query, mode=self.tokenizer_mode)
        ranked = sorted(self.engine.get_score_items(query_tokens), key=lambda item: item[1], reverse=True)
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk = self.chunks[idx]
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    domain=chunk.domain,
                    score=float(score),
                    source=source,
                    query=query,
                    evidence_text=chunk.text,
                    metadata={
                        "page": chunk.page,
                        "section": chunk.section,
                        "clause_id": chunk.clause_id,
                        "numbers": chunk.numbers,
                        "dates": chunk.dates,
                        "title": chunk.metadata.get("title", ""),
                    },
                )
            )
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path) -> None:
        """保存最小必要数据；BM25 引擎加载时重新构造。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "corpus_tokens": self.corpus_tokens,
                    "tokenizer_mode": self.tokenizer_mode,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25SearchIndex":
        """从磁盘恢复索引。"""
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            payload["chunks"],
            payload["corpus_tokens"],
            tokenizer_mode=payload.get("tokenizer_mode", "mixed"),
        )
