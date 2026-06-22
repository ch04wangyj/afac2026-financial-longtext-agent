"""V12 文档级高召回验证器。

该路径把 sparse retrieval 降级为候选排序器：每个选项都在题目指定文档内
独立检索，并补充精确数值/实体扫描和相邻上下文，再由 Qwen 统一裁决。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object, parse_answer
from agent.retrieve.claim_retrieval import build_claim_query_bundles
from agent.retrieve.claims import ClaimTarget, build_claim_targets, claim_to_retrieval_target
from agent.retrieve.evidence_selection import select_evidence_set
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import AnswerResult, Chunk, Question, RetrievalResult, TokenUsage


@dataclass(frozen=True)
class ExhaustiveVerifierConfig:
    """控制召回宽度和最终上下文预算。"""

    top_k_per_query: int = 10
    fused_top_k_per_claim: int = 64
    evidence_per_claim: int = 12
    evidence_chars_per_claim: int = 14_000
    max_context_chars: int = 52_000
    exact_sweep_per_doc: int = 8
    neighbor_anchors_per_doc: int = 2
    audit_enabled: bool = False
    answer_max_tokens: int = 2_048


class ExhaustiveVerifier:
    """执行选项级高召回、证据合并和统一答案裁决。"""

    def __init__(
        self,
        index: BM25SearchIndex,
        llm: QwenClient,
        config: ExhaustiveVerifierConfig | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.config = config or ExhaustiveVerifierConfig()

    def solve(self, question: Question) -> AnswerResult:
        """读取每个选项的平衡证据，并输出可直接提交的答案。"""
        evidence, retrieval_report = self.collect_evidence(question)
        context, evidence_map = _format_evidence_context(evidence, self.config.max_context_chars)
        evidence = evidence[: len(evidence_map)]
        messages = build_exhaustive_judge_messages(question, context)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            max_tokens=self.config.answer_max_tokens,
            enable_thinking=True,
        )
        total_usage = TokenUsage().add(response.usage)
        final_text = response.text
        audit_text = ""

        if self.config.audit_enabled:
            audit = self.llm.chat(
                build_exhaustive_audit_messages(question, context, response.text),
                temperature=0.0,
                max_tokens=self.config.answer_max_tokens,
                enable_thinking=True,
            )
            total_usage.add(audit.usage)
            audit_text = audit.text
            if _valid_answer(parse_answer(audit.text, question.answer_format), question):
                final_text = audit.text

        answer = parse_answer(final_text, question.answer_format)
        if not _valid_answer(answer, question):
            # 输出格式异常时再尝试首轮结果，仍失败才使用最低风险兜底。
            answer = parse_answer(response.text, question.answer_format)
        if not _valid_answer(answer, question):
            answer = sorted(question.options)[0]

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=_extract_confidence(final_text),
            evidence=evidence,
            token_usage=total_usage,
            raw_response=final_text,
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "exhaustive_document_verifier",
                "model": self.llm.settings.qwen_model,
                "audit_enabled": self.config.audit_enabled,
                "audit_response": audit_text,
                "retrieval_report": retrieval_report,
                "evidence_id_map": evidence_map,
            },
        )

    def collect_evidence(self, question: Question) -> tuple[list[RetrievalResult], dict]:
        """按选项和文档分别召回，防止一个文档或相似 chunk 垄断上下文。"""
        selected_by_claim: list[tuple[ClaimTarget, list[RetrievalResult], dict]] = []
        for claim in build_claim_targets(question):
            candidates = self._collect_claim_candidates(question, claim)
            selected, report = select_evidence_set(
                claim_to_retrieval_target(claim),
                candidates,
                top_k=self.config.evidence_per_claim,
                max_chars=self.config.evidence_chars_per_claim,
            )
            selected_by_claim.append(
                (
                    claim,
                    selected,
                    {
                        "candidate_count": len(candidates),
                        "selection": report.to_dict(),
                    },
                )
            )

        merged = _merge_claim_evidence(selected_by_claim, self.config.max_context_chars)
        return merged, {
            "strategy": "option_doc_balanced_exact_sweep",
            "max_context_chars": self.config.max_context_chars,
            "selected_count": len(merged),
            "selected_chars": sum(len(item.evidence_text or "") for item in merged),
            "claims": {
                claim.option_key: report
                for claim, _, report in selected_by_claim
            },
        }

    def _collect_claim_candidates(self, question: Question, claim: ClaimTarget) -> list[RetrievalResult]:
        bundles = build_claim_query_bundles(question, claim, max_bundles=8)
        ranked_lists: list[list[RetrievalResult]] = []
        weights: list[float] = []
        doc_scope = list(dict.fromkeys(question.doc_ids))

        for bundle in bundles:
            ranked_lists.append(
                self.index.search(
                    bundle.query,
                    top_k=self.config.top_k_per_query,
                    filter_doc_ids=set(doc_scope) or None,
                    source=f"exhaustive:{claim.option_key}:{bundle.intent}:mixed",
                    scoring_mode="bm25f_lite",
                )
            )
            weights.append(bundle.weight)
            # 每份指定文档独立召回，避免混合 Top-K 被长文档占满。
            for doc_id in doc_scope:
                ranked_lists.append(
                    self.index.search(
                        bundle.query,
                        top_k=self.config.top_k_per_query,
                        filter_doc_ids={doc_id},
                        source=f"exhaustive:{claim.option_key}:{bundle.intent}:{doc_id}",
                        scoring_mode="bm25f_lite",
                    )
                )
                weights.append(bundle.weight)

        fused = reciprocal_rank_fusion(
            ranked_lists,
            top_k=self.config.fused_top_k_per_claim,
            weights=weights,
        ) if ranked_lists else []
        exact = self._exact_document_sweep(claim, doc_scope)
        expanded = self._expand_anchor_context(claim, [*exact, *fused], doc_scope)
        return _dedupe_results([*exact, *fused, *expanded])

    def _exact_document_sweep(self, claim: ClaimTarget, doc_scope: list[str]) -> list[RetrievalResult]:
        """遍历指定文档，补回 BM25 排名可能漏掉的精确数字和实体命中。"""
        strong_terms = _strong_claim_terms(claim)
        exact_values = [value for value in claim.numbers if not _is_plain_year(value)]
        exact_dates = [value for value in claim.dates if len(_compact(value)) > 4]
        years = [value for value in claim.numbers if _is_plain_year(value)]
        output: list[RetrievalResult] = []
        for doc_id in doc_scope:
            scored: list[tuple[float, Chunk]] = []
            for chunk in self.index.doc_chunks_ordered.get(doc_id, []):
                text = _compact(chunk.text)
                if not text:
                    continue
                number_hits = sum(1 for value in exact_values if _compact(value) in text)
                date_hits = sum(1 for value in exact_dates if _compact(value) in text)
                year_hits = sum(1 for value in years if _compact(value) in text)
                term_hits = sum(1 for term in strong_terms if _compact(term) in text)
                # 普通年份在年报中出现过密，不能单独触发 exact sweep。
                if number_hits == 0 and date_hits == 0 and term_hits < 2:
                    continue
                structure_bonus = 0.8 if chunk.metadata.get("chunk_type") in {"table", "financial_metric_row"} else 0.0
                scored.append((number_hits * 4.0 + date_hits * 3.0 + min(1, year_hits) * 0.5 + term_hits + structure_bonus, chunk))
            scored.sort(key=lambda item: item[0], reverse=True)
            for score, chunk in scored[: self.config.exact_sweep_per_doc]:
                output.append(
                    self.index.result_from_chunk(
                        chunk,
                        # exact sweep 用于补候选，不应凭普通实体共现压过多查询 RRF。
                        score=0.25 + min(score, 10.0) * 0.01,
                        source=f"exhaustive:{claim.option_key}:exact_sweep",
                        query=claim.claim_text,
                    )
                )
        return output

    def _expand_anchor_context(
        self,
        claim: ClaimTarget,
        candidates: list[RetrievalResult],
        doc_scope: list[str],
    ) -> list[RetrievalResult]:
        """展开高分命中的同页和相邻块，恢复表头、例外条件和跨段上下文。"""
        output: list[RetrievalResult] = []
        for doc_id in doc_scope:
            anchors = [item for item in candidates if item.doc_id == doc_id][: self.config.neighbor_anchors_per_doc]
            for anchor in anchors:
                chunks = self.index.get_same_page_chunks(doc_id, anchor.metadata.get("page"), allowed_doc_ids={doc_id})
                chunks.extend(self.index.get_doc_neighbors(anchor.chunk_id, left=1, right=1, allowed_doc_ids={doc_id}))
                for chunk in chunks:
                    output.append(
                        self.index.result_from_chunk(
                            chunk,
                            score=max(0.01, float(anchor.score) * 0.9),
                            source=f"exhaustive:{claim.option_key}:context_expand",
                            query=claim.claim_text,
                        )
                    )
        return output


def build_exhaustive_judge_messages(question: Question, context: str) -> list[dict[str, str]]:
    """构造先判事实、再解释题干选择方向的统一裁决提示词。"""
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是金融长文档证据裁决器。只能依据给定证据作答。"
                "先逐项判断选项陈述本身的事实真伪，再解释题干要求选择正确项、错误项或最符合项。"
                "复合陈述任一子句不成立则整项不成立；比较题必须核对双方、年份、单位和口径；"
                "利润分配题必须区分末期/年末拟派方案与包含中期分红的全年合计，不能用全年合计否定末期方案；"
                "法规与合同题必须检查例外、否定词、主体和期限。禁止凭常识补全缺失证据。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题型：{question.answer_format}\n题干：{question.question}\n"
                f"选项：\n{options}\n\n证据：\n{context}\n\n"
                "返回单个 JSON 对象，不要 Markdown："
                '{"option_analysis":{"A":{"truth":"true|false|uncertain","citations":["E001"],"reason":"简短理由"}},'
                '"selection_rule":"题干要求选择什么","answer":"A或按字母排序的多选组合","confidence":0.0}'
            ),
        },
    ]


def build_exhaustive_audit_messages(question: Question, context: str, first_response: str) -> list[dict[str, str]]:
    """构造独立反证审计，重点检查常见 exact-match 失分模式。"""
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是独立审计员。检查初审是否漏选、错选，尤其核对否定题、复合断言、"
                "跨文档比较、表格列错位、年份和单位。不要因为初审给出某答案而默认同意。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题型：{question.answer_format}\n题干：{question.question}\n选项：\n{options}\n\n"
                f"证据：\n{context}\n\n初审：\n{first_response}\n\n"
                "返回单个 JSON 对象，不要 Markdown："
                '{"corrections":["发现的问题"],"answer":"A或按字母排序的多选组合","confidence":0.0}'
            ),
        },
    ]


def _strong_claim_terms(claim: ClaimTarget) -> list[str]:
    terms = [*claim.entities, *claim.must_terms, *claim.should_terms]
    output: list[str] = []
    for term in terms:
        compact = _compact(term)
        if len(compact) < 3 or compact in _GENERIC_TERMS:
            continue
        if compact not in output:
            output.append(compact)
    return output[:16]


def _merge_claim_evidence(
    selected_by_claim: list[tuple[ClaimTarget, list[RetrievalResult], dict]],
    max_chars: int,
) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[str] = set()
    used = 0
    # 轮询各选项，避免前两个选项先耗尽全局字符预算。
    max_len = max((len(items) for _, items, _ in selected_by_claim), default=0)
    for index in range(max_len):
        for _, items, _ in selected_by_claim:
            if index >= len(items):
                continue
            item = items[index]
            text_len = len(item.evidence_text or "")
            if item.chunk_id in seen or (output and used + text_len > max_chars):
                continue
            output.append(item)
            seen.add(item.chunk_id)
            used += text_len
    return output


def _format_evidence_context(
    evidence: list[RetrievalResult],
    max_chars: int,
) -> tuple[str, dict[str, str]]:
    blocks: list[str] = []
    evidence_map: dict[str, str] = {}
    used = 0
    for index, item in enumerate(evidence, start=1):
        evidence_id = f"E{index:03d}"
        header = (
            f"[{evidence_id}] doc={item.doc_id} page={item.metadata.get('page')} "
            f"section={item.metadata.get('section', '')} chunk={item.chunk_id}"
        )
        block = f"{header}\n{item.evidence_text.strip()}"
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        evidence_map[evidence_id] = item.chunk_id
        used += len(block)
    return "\n\n".join(blocks), evidence_map


def _valid_answer(answer: str, question: Question) -> bool:
    if not answer or any(letter not in question.options for letter in answer):
        return False
    if question.answer_format in {"mcq", "tf"}:
        return len(answer) == 1
    return answer == "".join(sorted(set(answer)))


def _extract_confidence(text: str) -> float:
    obj = extract_json_object(text) or {}
    value = obj.get("confidence", 0.0)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _dedupe_results(items: list[RetrievalResult]) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[str] = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        output.append(item)
        seen.add(item.chunk_id)
    return output


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).replace("，", ",").replace("％", "%")


def _is_plain_year(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}年?", _compact(value)))


_GENERIC_TERMS = {
    "以下哪些",
    "以下哪个",
    "下列说法",
    "正确的是",
    "错误的是",
    "判断选项",
    "是否正确",
}
