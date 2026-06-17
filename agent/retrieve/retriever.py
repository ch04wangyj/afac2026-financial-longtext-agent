"""默认在线检索器，供真实答题链路调用。"""

from __future__ import annotations

from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.reasoning.logicrag import build_logicrag_rrf_queries
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.query import build_rule_queries
from agent.schemas import Question, RetrievalResult


class Retriever:
    """封装单路 question_options 和多路规则 RRF 检索。"""

    def __init__(
        self,
        index: BM25SearchIndex,
        doc_index: DocumentSearchIndex | None = None,
        top_k_per_query: int = 20,
        fused_top_k: int = 30,
        strategy: str = "hybrid",
        blind_top_docs: int = 8,
    ) -> None:
        self.index = index
        self.doc_index = doc_index
        self.top_k_per_query = top_k_per_query
        self.fused_top_k = fused_top_k
        self.strategy = strategy
        self.blind_top_docs = blind_top_docs

    def retrieve(self, question: Question, restrict_to_doc_ids: bool = True) -> list[RetrievalResult]:
        """检索题目相关证据；A 组默认可按题目 doc_ids 限定候选文档。"""
        filter_doc_ids = self._candidate_doc_filter(question, restrict_to_doc_ids)
        if self.strategy == "question_options":
            return self.index.search(
                query=self._question_with_options(question),
                top_k=self.fused_top_k,
                filter_doc_ids=filter_doc_ids,
                source="question_options",
            )

        ranked_lists: list[list[RetrievalResult]] = []
        if self.strategy == "hybrid":
            # 主查询使用“题干+选项”，实测 A 组 doc 命中最稳。
            ranked_lists.append(
                self.index.search(
                    query=self._question_with_options(question),
                    top_k=self.top_k_per_query,
                    filter_doc_ids=filter_doc_ids,
                    source="question_options",
                )
            )

        queries = build_rule_queries(question)
        if self.strategy in {"logicrag", "logicrag_agent"}:
            queries = build_logicrag_rrf_queries(question)

        for query in queries:
            ranked_lists.append(
                self.index.search(
                    query=query,
                    top_k=self.top_k_per_query,
                    filter_doc_ids=filter_doc_ids,
                    source="bm25",
                )
            )
        return reciprocal_rank_fusion(ranked_lists, top_k=self.fused_top_k)

    @staticmethod
    def _question_with_options(question: Question) -> str:
        """拼接题干和选项，避免选项实体未出现在题干时漏召回。"""
        return f"{question.question} " + " ".join(
            f"{key} {value}" for key, value in sorted(question.options.items())
        )

    def _candidate_doc_filter(self, question: Question, restrict_to_doc_ids: bool) -> set[str] | None:
        """A 榜用题目 doc_ids；B 榜无 doc_ids 时用文档级 BM25 盲搜候选。"""
        if restrict_to_doc_ids and question.doc_ids:
            return set(question.doc_ids)
        if self.doc_index and (not question.doc_ids or not restrict_to_doc_ids):
            doc_ids = self.doc_index.search_doc_ids(
                self._question_with_options(question),
                top_k=self.blind_top_docs,
                domain=question.domain,
            )
            return set(doc_ids) if doc_ids else None
        return None
