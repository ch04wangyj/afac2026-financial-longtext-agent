"""轻量 BM25 检索实现。
当前项目直接使用内置纯 Python sparse retrieval，不依赖额外 BM25 第三方引擎。
索引只使用词法特征，不使用 embedding。
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from agent.preprocess.chunkers import extract_dates, extract_numbers
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
        return [self.score(query_tokens, idx) for idx in range(self.doc_count)]

    def get_score_items(self, query_tokens: list[str]) -> list[tuple[int, float]]:
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
        unique = set(query_tokens)
        if len(unique) <= self.max_query_terms:
            return list(unique)
        return sorted(unique, key=lambda token: self.idf.get(token, 0.0), reverse=True)[: self.max_query_terms]


class BM25SearchIndex:
    """可持久化的 chunk 检索索引。"""

    FIELD_AWARE_WEIGHTS: dict[str, float] = {
        "base": 1.0,
        "title": 0.30,
        "section": 0.2,
        "clause_id": 0.30,
        "numbers": 0.35,
        "dates": 0.3,
        "caption": 0.15,
        "structured": 0.55,
    }

    def __init__(
        self,
        chunks: list[Chunk],
        corpus_tokens: list[list[str]] | None = None,
        tokenizer_mode: str = "mixed",
        parent_chunks: list[Chunk] | None = None,
    ) -> None:
        self.chunks = chunks
        # V3 只对子块建倒排索引；父块单独保存，避免粗细粒度文本互相争抢 Top-K。
        self.parent_chunks = list(parent_chunks or [])
        self.parent_chunk_by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.parent_chunks}
        self.tokenizer_mode = tokenizer_mode
        self.default_search_mode = "bm25"
        self.corpus_tokens = corpus_tokens or [tokenize_chunk(chunk, mode=tokenizer_mode) for chunk in chunks]
        self.engine = SimpleBM25(self.corpus_tokens)
        self.field_token_corpora = self._build_field_token_corpora()
        self.field_engines = {
            field: SimpleBM25(tokens, max_query_terms=32)
            for field, tokens in self.field_token_corpora.items()
        }
        self.chunk_by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.doc_chunks_ordered: dict[str, list[Chunk]] = defaultdict(list)
        self.doc_chunk_pos: dict[str, int] = {}
        self.page_to_chunk_ids: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.clause_to_chunk_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.section_to_chunk_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        for chunk in self.chunks:
            self.doc_chunks_ordered[chunk.doc_id].append(chunk)
        for doc_id, doc_chunks in self.doc_chunks_ordered.items():
            page_counts: dict[int, int] = defaultdict(int)
            for pos, chunk in enumerate(doc_chunks):
                self.doc_chunk_pos[chunk.chunk_id] = pos
                chunk.metadata.setdefault("doc_seq", pos)
                if chunk.page is not None:
                    chunk.metadata.setdefault("page_chunk_idx", page_counts[chunk.page])
                    page_counts[chunk.page] += 1
                    self.page_to_chunk_ids[(doc_id, int(chunk.page))].append(chunk.chunk_id)
                clause = str(chunk.clause_id or "").strip()
                if clause:
                    self.clause_to_chunk_ids[(doc_id, clause)].append(chunk.chunk_id)
                section_key = self._normalize_section(chunk.section)
                if section_key:
                    self.section_to_chunk_ids[(doc_id, section_key)].append(chunk.chunk_id)

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        tokenizer_mode: str = "mixed",
        parent_chunks: list[Chunk] | None = None,
    ) -> "BM25SearchIndex":
        return cls(chunks, tokenizer_mode=tokenizer_mode, parent_chunks=parent_chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
        filter_chunk_types: set[str] | None = None,
        source: str = "bm25",
        scoring_mode: str | None = None,
    ) -> list[RetrievalResult]:
        scoring_mode = scoring_mode or self.default_search_mode
        if scoring_mode == "bm25f_lite":
            return self._field_aware_search(
                query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
                filter_chunk_types=filter_chunk_types,
                source=source,
            )

        query_tokens = tokenize(query, mode=self.tokenizer_mode)
        ranked = sorted(self.engine.get_score_items(query_tokens), key=lambda item: item[1], reverse=True)
        results: list[RetrievalResult] = []
        for idx, score in ranked:
            if score <= 0:
                break
            chunk = self.chunks[idx]
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue
            if filter_chunk_types and str(chunk.metadata.get("chunk_type", "text")) not in filter_chunk_types:
                continue
            results.append(self.result_from_chunk(chunk, score=float(score), source=source, query=query))
            if len(results) >= top_k:
                break
        return results

    def _field_aware_search(
        self,
        query: str,
        top_k: int,
        filter_doc_ids: set[str] | None,
        filter_chunk_types: set[str] | None,
        source: str,
    ) -> list[RetrievalResult]:
        query_tokens = tokenize(query, mode=self.tokenizer_mode)
        if not query_tokens:
            return []

        query_numbers = extract_numbers(query)
        query_dates = extract_dates(query)
        component_scores = self._collect_component_scores(query_tokens, query_numbers=query_numbers, query_dates=query_dates)
        if not component_scores:
            return []

        candidate_scores = self._filter_component_scores(
            component_scores,
            filter_doc_ids,
            filter_chunk_types,
        )
        if not candidate_scores:
            return []

        normalized_scores = self._normalize_component_scores(candidate_scores)
        ranked: list[tuple[int, float]] = []
        for idx, parts in normalized_scores.items():
            chunk = self.chunks[idx]
            total = sum(parts.get(field, 0.0) * self.FIELD_AWARE_WEIGHTS.get(field, 0.0) for field in self.FIELD_AWARE_WEIGHTS)
            if total <= 0:
                continue
            ranked.append((idx, float(total)))

        ranked.sort(key=lambda item: item[1], reverse=True)
        results: list[RetrievalResult] = []
        for idx, total_score in ranked[:top_k]:
            chunk = self.chunks[idx]
            result = self.result_from_chunk(chunk, score=float(total_score), source=source, query=query)
            raw_parts = component_scores.get(idx, {})
            norm_parts = normalized_scores.get(idx, {})
            matched_fields = [
                field for field, score in norm_parts.items() if field != "base" and score > 0
            ]
            result.metadata["score_breakdown"] = {
                "mode": "bm25f_lite",
                "weights": dict(self.FIELD_AWARE_WEIGHTS),
                "raw": {field: round(float(raw_parts.get(field, 0.0)), 6) for field in self.FIELD_AWARE_WEIGHTS},
                "normalized": {field: round(float(norm_parts.get(field, 0.0)), 6) for field in self.FIELD_AWARE_WEIGHTS},
                "matched_fields": matched_fields,
                "total": round(float(total_score), 6),
            }
            results.append(result)
        return results

    def _collect_component_scores(
        self,
        query_tokens: list[str],
        *,
        query_numbers: list[str],
        query_dates: list[str],
    ) -> dict[int, dict[str, float]]:
        combined: dict[int, dict[str, float]] = defaultdict(dict)
        for idx, score in self.engine.get_score_items(query_tokens):
            if score > 0:
                combined[idx]["base"] = float(score)

        for field in ("title", "section", "clause_id", "caption", "structured"):
            for idx, score in self.field_engines[field].get_score_items(query_tokens):
                if score > 0:
                    combined[idx][field] = float(score)

        if query_numbers:
            number_tokens = tokenize(" ".join(query_numbers), mode=self.tokenizer_mode)
            for idx, score in self.field_engines["numbers"].get_score_items(number_tokens):
                if score > 0:
                    combined[idx]["numbers"] = float(score)

        if query_dates:
            date_tokens = tokenize(" ".join(query_dates), mode=self.tokenizer_mode)
            for idx, score in self.field_engines["dates"].get_score_items(date_tokens):
                if score > 0:
                    combined[idx]["dates"] = float(score)
        return combined

    def _filter_component_scores(
        self,
        component_scores: dict[int, dict[str, float]],
        filter_doc_ids: set[str] | None,
        filter_chunk_types: set[str] | None = None,
    ) -> dict[int, dict[str, float]]:
        if not filter_doc_ids and not filter_chunk_types:
            return component_scores
        return {
            idx: parts
            for idx, parts in component_scores.items()
            if (not filter_doc_ids or self.chunks[idx].doc_id in filter_doc_ids)
            and (
                not filter_chunk_types
                or str(self.chunks[idx].metadata.get("chunk_type", "text")) in filter_chunk_types
            )
        }

    def _normalize_component_scores(self, component_scores: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
        maxima = {
            field: max((parts.get(field, 0.0) for parts in component_scores.values()), default=0.0)
            for field in self.FIELD_AWARE_WEIGHTS
        }
        normalized: dict[int, dict[str, float]] = {}
        for idx, parts in component_scores.items():
            normalized[idx] = {
                field: (parts.get(field, 0.0) / maxima[field]) if maxima[field] > 0 else 0.0
                for field in self.FIELD_AWARE_WEIGHTS
            }
        return normalized

    def _build_field_token_corpora(self) -> dict[str, list[list[str]]]:
        corpora = {field: [] for field in self.FIELD_AWARE_WEIGHTS if field != "base"}
        for chunk in self.chunks:
            corpora["title"].append(tokenize(str(chunk.metadata.get("title", "")), mode=self.tokenizer_mode))
            corpora["section"].append(tokenize(chunk.section, mode=self.tokenizer_mode))
            corpora["clause_id"].append(tokenize(chunk.clause_id, mode=self.tokenizer_mode))
            corpora["numbers"].append(tokenize(" ".join(chunk.numbers), mode=self.tokenizer_mode))
            corpora["dates"].append(tokenize(" ".join(chunk.dates), mode=self.tokenizer_mode))
            corpora["caption"].append(tokenize(str(chunk.metadata.get("caption", "")), mode=self.tokenizer_mode))
            corpora["structured"].append(
                tokenize(
                    " ".join(str(item) for item in chunk.metadata.get("extra_index_fields", []) if str(item).strip()),
                    mode=self.tokenizer_mode,
                )
            )
        return corpora

    def result_from_chunk(self, chunk: Chunk, *, score: float, source: str, query: str) -> RetrievalResult:
        return RetrievalResult(
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
                "chunk_type": chunk.metadata.get("chunk_type", "text"),
                "caption": chunk.metadata.get("caption", ""),
                "extra_index_fields": chunk.metadata.get("extra_index_fields", []),
                "financial_row": chunk.metadata.get("financial_row", {}),
                "parent_chunk_id": chunk.metadata.get("parent_chunk_id", ""),
                "parser_name": chunk.metadata.get("parser_name", ""),
                "layout_source": chunk.metadata.get("layout_source", ""),
                "table_id": chunk.metadata.get("table_id", ""),
                "table_header": chunk.metadata.get("table_header", ""),
                "table_unit": chunk.metadata.get("table_unit", ""),
                "table_continuation": chunk.metadata.get("table_continuation", False),
                "doc_seq": chunk.metadata.get("doc_seq", self.doc_chunk_pos.get(chunk.chunk_id, 0)),
                "page_chunk_idx": chunk.metadata.get("page_chunk_idx", 0),
            },
        )

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunk_by_id.get(chunk_id)

    def get_parent_chunk(self, chunk_id: str) -> Chunk | None:
        """按子块 metadata.parent_chunk_id 读取未参与检索的父块。"""
        chunk = self.get_chunk(chunk_id)
        parent_id = str(chunk.metadata.get("parent_chunk_id", "")) if chunk else str(chunk_id)
        return self.parent_chunk_by_id.get(parent_id)

    def get_doc_neighbors(
        self,
        chunk_id: str,
        *,
        left: int = 1,
        right: int = 1,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[Chunk]:
        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            return []
        if allowed_doc_ids and chunk.doc_id not in allowed_doc_ids:
            return []
        doc_chunks = self.doc_chunks_ordered.get(chunk.doc_id, [])
        pos = self.doc_chunk_pos.get(chunk_id, 0)
        start = max(0, pos - left)
        end = min(len(doc_chunks), pos + right + 1)
        return [item for item in doc_chunks[start:end] if not allowed_doc_ids or item.doc_id in allowed_doc_ids]

    def get_same_page_chunks(self, doc_id: str, page: int | None, *, allowed_doc_ids: set[str] | None = None) -> list[Chunk]:
        if page is None:
            return []
        if allowed_doc_ids and doc_id not in allowed_doc_ids:
            return []
        return [self.chunk_by_id[chunk_id] for chunk_id in self.page_to_chunk_ids.get((doc_id, int(page)), [])]

    def get_same_clause_chunks(self, doc_id: str, clause_id: str, *, allowed_doc_ids: set[str] | None = None) -> list[Chunk]:
        clause_id = str(clause_id or "").strip()
        if not clause_id:
            return []
        if allowed_doc_ids and doc_id not in allowed_doc_ids:
            return []
        return [self.chunk_by_id[chunk_id] for chunk_id in self.clause_to_chunk_ids.get((doc_id, clause_id), [])]

    def get_same_section_chunks(self, doc_id: str, section: str, *, allowed_doc_ids: set[str] | None = None) -> list[Chunk]:
        section_key = self._normalize_section(section)
        if not section_key:
            return []
        if allowed_doc_ids and doc_id not in allowed_doc_ids:
            return []
        return [self.chunk_by_id[chunk_id] for chunk_id in self.section_to_chunk_ids.get((doc_id, section_key), [])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "corpus_tokens": self.corpus_tokens,
                    "tokenizer_mode": self.tokenizer_mode,
                    "parent_chunks": self.parent_chunks,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "BM25SearchIndex":
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            payload["chunks"],
            payload["corpus_tokens"],
            tokenizer_mode=payload.get("tokenizer_mode", "mixed"),
            parent_chunks=payload.get("parent_chunks", []),
        )

    @staticmethod
    def _normalize_section(section: str) -> str:
        return " ".join(str(section or "").split()).lower()
