"""无 embedding 的文档结构导航索引。

该模块借鉴 PageIndex/BookRAG 的“先定位自然结构，再读取局部证据”思想，
但不调用模型生成目录，也不替换原有 BM25 召回。导航结果只用于补充候选页和
相邻页，避免结构路由出错时压掉 V4 已经能够召回的证据。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from agent.index.bm25 import BM25SearchIndex, SimpleBM25
from agent.index.tokenizer import tokenize
from agent.schemas import Chunk


@dataclass(frozen=True)
class NavigationNode:
    """一个可导航的页、章节或无页码滑动窗口。"""

    node_id: str
    doc_id: str
    domain: str
    page: int | None
    section: str
    chunk_ids: tuple[str, ...]
    search_text: str


@dataclass(frozen=True)
class NavigationHit:
    """结构导航命中及其稀疏相关度。"""

    node_id: str
    doc_id: str
    page: int | None
    section: str
    score: float
    rank_in_doc: int


class StructureNavigator:
    """对页/章节描述建立小型 BM25，提供可解释的两阶段导航。"""

    def __init__(
        self,
        index: BM25SearchIndex,
        *,
        max_node_chars: int = 8_000,
        fallback_window_size: int = 10,
    ) -> None:
        self.index = index
        self.max_node_chars = max(1_000, int(max_node_chars))
        self.fallback_window_size = max(4, int(fallback_window_size))
        self.nodes = self._build_nodes(index.chunks)
        self.node_by_id = {node.node_id: node for node in self.nodes}
        corpus = [tokenize(node.search_text, mode=index.tokenizer_mode) for node in self.nodes]
        self.engine = SimpleBM25(corpus, max_query_terms=96)

    def search(
        self,
        query: str,
        *,
        doc_ids: Iterable[str],
        top_k_per_doc: int = 3,
    ) -> list[NavigationHit]:
        """在每份指定文档内保留若干最佳结构节点。"""
        doc_order = list(dict.fromkeys(doc_ids))
        allowed = set(doc_order)
        if not allowed or not query.strip():
            return []
        query_tokens = tokenize(query, mode=self.index.tokenizer_mode)
        if not query_tokens:
            return []

        grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for node_index, score in self.engine.get_score_items(query_tokens):
            node = self.nodes[node_index]
            if score > 0 and node.doc_id in allowed:
                grouped[node.doc_id].append((float(score), node_index))

        output: list[NavigationHit] = []
        for doc_id in doc_order:
            ranked = sorted(grouped.get(doc_id, []), key=lambda item: (-item[0], item[1]))
            for rank, (score, node_index) in enumerate(ranked[:top_k_per_doc], start=1):
                node = self.nodes[node_index]
                output.append(
                    NavigationHit(
                        node_id=node.node_id,
                        doc_id=node.doc_id,
                        page=node.page,
                        section=node.section,
                        score=score,
                        rank_in_doc=rank,
                    )
                )
        return output

    def expand_chunks(
        self,
        hits: Iterable[NavigationHit],
        *,
        page_radius: int = 1,
        allowed_chunk_types: set[str] | None = None,
    ) -> list[tuple[NavigationHit, Chunk]]:
        """展开命中节点及相邻页，结果按首次命中去重。"""
        output: list[tuple[NavigationHit, Chunk]] = []
        seen: set[str] = set()
        for hit in hits:
            node = self.node_by_id.get(hit.node_id)
            if node is None:
                continue
            chunk_ids = list(node.chunk_ids)
            if node.page is not None:
                for page in range(max(1, node.page - page_radius), node.page + page_radius + 1):
                    chunk_ids.extend(self.index.page_to_chunk_ids.get((node.doc_id, page), []))
            for chunk_id in chunk_ids:
                if chunk_id in seen:
                    continue
                chunk = self.index.get_chunk(chunk_id)
                if chunk is None:
                    continue
                chunk_type = str(chunk.metadata.get("chunk_type", "text"))
                if allowed_chunk_types and chunk_type not in allowed_chunk_types:
                    continue
                seen.add(chunk_id)
                output.append((hit, chunk))
        return output

    def _build_nodes(self, chunks: list[Chunk]) -> list[NavigationNode]:
        grouped: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.page is not None:
                key = (chunk.doc_id, f"page:{int(chunk.page)}")
            else:
                section = _normalize(chunk.section)
                if section:
                    key = (chunk.doc_id, f"section:{section}")
                else:
                    sequence = int(chunk.metadata.get("doc_seq", 0) or 0)
                    key = (chunk.doc_id, f"window:{sequence // self.fallback_window_size}")
            grouped[key].append(chunk)

        nodes: list[NavigationNode] = []
        for (doc_id, group_key), members in grouped.items():
            first = members[0]
            page = first.page if group_key.startswith("page:") else None
            section = _best_section(members)
            search_text = _compose_search_text(members, self.max_node_chars)
            digest = hashlib.sha1(f"{doc_id}:{group_key}".encode("utf-8")).hexdigest()[:14]
            nodes.append(
                NavigationNode(
                    node_id=f"nav:{digest}",
                    doc_id=doc_id,
                    domain=first.domain,
                    page=page,
                    section=section,
                    chunk_ids=tuple(chunk.chunk_id for chunk in members),
                    search_text=search_text,
                )
            )
        nodes.sort(key=lambda node: (node.domain, node.doc_id, node.page or 0, node.node_id))
        return nodes


def _compose_search_text(chunks: list[Chunk], max_chars: int) -> str:
    """组合标题、结构字段和正文；按规范化文本去重以抑制 V3/V4 重复块。"""
    structural: list[str] = []
    bodies: list[str] = []
    seen_structural: set[str] = set()
    seen_bodies: set[str] = set()

    def add(target: list[str], seen: set[str], value: object) -> None:
        text = " ".join(str(value or "").split())
        key = _normalize(text)
        if text and key not in seen:
            seen.add(key)
            target.append(text)

    for chunk in chunks:
        add(structural, seen_structural, chunk.metadata.get("title", ""))
        add(structural, seen_structural, chunk.section)
        add(structural, seen_structural, chunk.clause_id)
        add(structural, seen_structural, chunk.metadata.get("caption", ""))
        add(structural, seen_structural, chunk.metadata.get("table_header", ""))
        add(structural, seen_structural, chunk.metadata.get("table_unit", ""))
        for field in chunk.metadata.get("extra_index_fields", []):
            add(structural, seen_structural, field)
        add(bodies, seen_bodies, chunk.text)

    prefix = "\n".join(structural)
    remaining = max(0, max_chars - len(prefix) - 1)
    body = "\n".join(bodies)
    return f"{prefix}\n{body[:remaining]}".strip()


def _best_section(chunks: list[Chunk]) -> str:
    sections = [" ".join(chunk.section.split()) for chunk in chunks if chunk.section.strip()]
    return max(sections, key=len, default="")


def _normalize(value: str) -> str:
    return "".join(str(value or "").split()).casefold()
