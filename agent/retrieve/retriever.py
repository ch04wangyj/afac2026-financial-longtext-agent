"""默认在线检索器，供真实答题链路调用。"""

from __future__ import annotations

from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.reasoning.logicrag import build_logicrag_rrf_queries
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.query import build_rule_queries
from agent.retrieve.targets import question_with_options
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
        self.index.default_search_mode = "bm25"

    def retrieve(self, question: Question, restrict_to_doc_ids: bool = True) -> list[RetrievalResult]:
        filter_doc_ids = self._candidate_doc_filter(question, restrict_to_doc_ids)
        question_options = self._question_with_options(question)
        scoring_mode = "bm25f_lite" if self.strategy == "bm25f_lite_rrf" else None
        if self.strategy == "question_options":
            return self.index.search(
                query=question_options,
                top_k=self.fused_top_k,
                filter_doc_ids=filter_doc_ids,
                source="question_options",
                scoring_mode=scoring_mode,
            )

        if self.strategy == "bm25f_lite_rrf":
            ranked_lists = [
                self.index.search(
                    query=query,
                    top_k=self.top_k_per_query,
                    filter_doc_ids=filter_doc_ids,
                    source="bm25f_lite_rrf",
                    scoring_mode="bm25f_lite",
                )
                for query in build_rule_queries(question)
            ]
            return reciprocal_rank_fusion(ranked_lists, top_k=self.fused_top_k)

        ranked_lists: list[list[RetrievalResult]] = []
        if self.strategy == "hybrid":
            ranked_lists.append(
                self.index.search(
                    query=question_options,
                    top_k=self.top_k_per_query,
                    filter_doc_ids=filter_doc_ids,
                    source="question_options",
                )
            )

        queries = build_rule_queries(question)
        if self.strategy in {"logicrag", "logicrag_agent"}:
            seed_results = self.index.search(
                query=question_options,
                top_k=self.top_k_per_query,
                filter_doc_ids=filter_doc_ids,
                source="logicrag_seed",
            )
            ranked_lists.append(seed_results)
            queries = build_logicrag_rrf_queries(question, seed_results=seed_results)

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
        return question_with_options(question)

    def _candidate_doc_filter(self, question: Question, restrict_to_doc_ids: bool) -> set[str] | None:
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
