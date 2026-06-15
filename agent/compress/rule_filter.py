"""规则式证据压缩。

V1 不用额外压缩模型，而是靠关键词、数字、日期和条款号保留高价值证据。
"""

from __future__ import annotations

from agent.index.tokenizer import tokenize
from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.schemas import Question, RetrievalResult


class RuleEvidenceCompressor:
    """从检索结果中挑选有限长度的证据片段。"""

    def __init__(self, max_chars: int = 6000, top_k: int = 8) -> None:
        self.max_chars = max_chars
        self.top_k = top_k

    def compress(self, question: Question, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """按规则分数排序，并控制总字符预算。

        多文档题先保证每个候选文档至少有一条证据，避免只把最高分文档送给 Qwen。
        """
        scored = [(self._score(question, result), result) for result in results]
        ranked_pairs = sorted(scored, key=lambda item: item[0], reverse=True)
        ranked = [result for _, result in ranked_pairs]
        selected: list[RetrievalResult] = []
        selected_ids: set[str] = set()
        used_chars = 0

        doc_quota = list(dict.fromkeys(question.doc_ids))
        if len(doc_quota) > 1:
            best_by_doc: dict[str, RetrievalResult] = {}
            for _, result in ranked_pairs:
                if result.doc_id in doc_quota and result.doc_id not in best_by_doc:
                    best_by_doc[result.doc_id] = result
            for doc_id in doc_quota:
                result = best_by_doc.get(doc_id)
                if result is None:
                    continue
                used_chars = self._try_select(result, selected, selected_ids, used_chars)
                if len(selected) >= self.top_k:
                    return selected

        for result in ranked:
            used_chars = self._try_select(result, selected, selected_ids, used_chars)
            if len(selected) >= self.top_k:
                break
        return selected

    def _try_select(
        self,
        result: RetrievalResult,
        selected: list[RetrievalResult],
        selected_ids: set[str],
        used_chars: int,
    ) -> int:
        """尝试加入一条证据，负责去重和字符预算控制。"""
        if result.chunk_id in selected_ids:
            return used_chars
        text = result.evidence_text.strip()
        if not text:
            return used_chars
        if used_chars + len(text) > self.max_chars and selected:
            return used_chars
        selected.append(result)
        selected_ids.add(result.chunk_id)
        return used_chars + len(text)

    def _score(self, question: Question, result: RetrievalResult) -> float:
        """计算证据对题目的相关性，重点奖励数字/日期/条款命中。"""
        q_text = f"{question.question} {' '.join(question.options.values())}"
        q_terms = set(tokenize(q_text, use_jieba=False))
        e_terms = set(tokenize(result.evidence_text, use_jieba=False))
        overlap = len(q_terms & e_terms)
        q_numbers = set(extract_numbers(q_text))
        q_dates = set(extract_dates(q_text))
        e_numbers = set(result.metadata.get("numbers", []))
        e_dates = set(result.metadata.get("dates", []))
        clause_bonus = 2.0 if result.metadata.get("clause_id") else 0.0
        number_bonus = 3.0 * len(q_numbers & e_numbers)
        date_bonus = 3.0 * len(q_dates & e_dates)
        return result.score + overlap * 0.1 + clause_bonus + number_bonus + date_bonus
