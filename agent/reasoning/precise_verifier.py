"""V13 精确证据验证器：原子子块检索、谓词真实值召回和紧凑裁决。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.data.document_aliases import document_label, option_doc_scope
from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object, parse_answer
from agent.retrieve.claims import ClaimTarget, build_claim_targets
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.structured_queries import extract_query_entities
from agent.retrieve.structure_navigator import NavigationHit, StructureNavigator
from agent.retrieve.verification_queries import (
    build_verification_query_bundles,
    extract_candidate_values,
    extract_predicate_terms,
)
from agent.retrieve.verification_rerank import select_verification_evidence
from agent.schemas import AnswerResult, Chunk, Question, RetrievalResult, TokenUsage


@dataclass(frozen=True)
class PreciseVerifierConfig:
    """V13 默认预算显著小于 V12，减少无关证据和 Token 惩罚。"""

    top_k_per_query: int = 12
    fused_top_k_per_claim: int = 48
    predicate_scan_per_doc: int = 8
    evidence_per_claim: int = 4
    evidence_chars_per_claim: int = 3_200
    max_context_chars: int = 12_000
    answer_max_tokens: int = 1_024
    enable_thinking: bool = True
    audit_enabled: bool = False
    enable_structure_navigation: bool = False
    assemble_answer_from_checks: bool = False
    navigation_nodes_per_doc: int = 3
    navigation_candidates_per_doc: int = 10
    navigation_page_radius: int = 1
    strategy_name: str = "v13_precise_verifier"
    search_chunk_types: tuple[str, ...] = (
        "atomic_text",
        "table_row",
        "financial_metric_row",
        "figure",
        "layout_text",
        "layout_table_row",
    )


class PreciseVerifier:
    """用支持/反证平衡证据一次性裁决所有选项。"""

    def __init__(
        self,
        index: BM25SearchIndex,
        llm: QwenClient,
        config: PreciseVerifierConfig | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.config = config or PreciseVerifierConfig()
        self.structure_navigator = (
            StructureNavigator(index) if self.config.enable_structure_navigation else None
        )

    def solve(self, question: Question) -> AnswerResult:
        evidence, report = self.collect_evidence(question)
        context, evidence_map = _format_grouped_context(question, evidence, self.config.max_context_chars)
        response = self.llm.chat(
            build_precise_judge_messages(question, context),
            temperature=0.0,
            max_tokens=self.config.answer_max_tokens,
            enable_thinking=self.config.enable_thinking,
        )
        usage = TokenUsage().add(response.usage)
        final_text = response.text
        audit_text = ""

        if self.config.audit_enabled:
            audit = self.llm.chat(
                build_precise_audit_messages(question, context, response.text),
                temperature=0.0,
                max_tokens=min(768, self.config.answer_max_tokens),
                enable_thinking=self.config.enable_thinking,
            )
            usage.add(audit.usage)
            audit_text = audit.text
            audited_answer = parse_answer(audit.text, question.answer_format)
            if _valid_answer(audited_answer, question):
                final_text = audit.text

        answer = (
            _answer_from_checks(final_text, question)
            if self.config.assemble_answer_from_checks
            else ""
        )
        if not _valid_answer(answer, question):
            answer = parse_answer(final_text, question.answer_format)
        if not _valid_answer(answer, question):
            answer = parse_answer(response.text, question.answer_format)
        if not _valid_answer(answer, question):
            answer = sorted(question.options)[0]

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=_extract_confidence(final_text),
            evidence=evidence,
            token_usage=usage,
            raw_response=final_text,
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": self.config.strategy_name,
                "model": self.llm.settings.qwen_model,
                "audit_enabled": self.config.audit_enabled,
                "audit_response": audit_text,
                "answer_assembly": (
                    "checks_truth" if self.config.assemble_answer_from_checks else "model_answer"
                ),
                "retrieval_report": report,
                "evidence_id_map": evidence_map,
            },
        )

    def collect_evidence(self, question: Question) -> tuple[list[RetrievalResult], dict]:
        selected_by_claim: list[tuple[ClaimTarget, list[RetrievalResult], dict]] = []
        for claim in build_claim_targets(question):
            predicates = extract_predicate_terms(question, claim)
            candidates = self._collect_claim_candidates(question, claim, predicates)
            selected, selection_report = select_verification_evidence(
                claim,
                candidates,
                predicates,
                top_k=self.config.evidence_per_claim,
                max_chars=self.config.evidence_chars_per_claim,
            )
            selected = [self._restore_parent_excerpt(claim, item) for item in selected]
            for item in selected:
                item.metadata["option_key"] = claim.option_key
            selected_by_claim.append(
                (
                    claim,
                    selected,
                    {
                        "predicate_terms": predicates,
                        "candidate_values": extract_candidate_values(claim),
                        "selection": selection_report.to_dict(),
                    },
                )
            )

        merged = _merge_claim_evidence(selected_by_claim, self.config.max_context_chars)
        return merged, {
            "strategy": self.config.strategy_name,
            "selected_count": len(merged),
            "selected_chars": sum(len(item.evidence_text or "") for item in merged),
            "max_context_chars": self.config.max_context_chars,
            "claims": {claim.option_key: report for claim, _, report in selected_by_claim},
        }

    def _collect_claim_candidates(
        self,
        question: Question,
        claim: ClaimTarget,
        predicates: list[str],
    ) -> list[RetrievalResult]:
        bundles = build_verification_query_bundles(question, claim)
        doc_scope = list(dict.fromkeys(option_doc_scope(question, claim.option_text)))
        ranked_lists: list[list[RetrievalResult]] = []
        weights: list[float] = []
        chunk_types = set(self.config.search_chunk_types)

        for bundle in bundles:
            ranked_lists.append(
                self.index.search(
                    bundle.query,
                    top_k=self.config.top_k_per_query,
                    filter_doc_ids=set(doc_scope) or None,
                    filter_chunk_types=chunk_types,
                    source=f"{self.config.strategy_name}:{claim.option_key}:{bundle.intent}:mixed",
                    scoring_mode="bm25f_lite",
                )
            )
            weights.append(bundle.weight)
            for doc_id in doc_scope:
                ranked_lists.append(
                    self.index.search(
                        bundle.query,
                        top_k=self.config.top_k_per_query,
                        filter_doc_ids={doc_id},
                        filter_chunk_types=chunk_types,
                        source=f"{self.config.strategy_name}:{claim.option_key}:{bundle.intent}:{doc_id}",
                        scoring_mode="bm25f_lite",
                    )
                )
                weights.append(bundle.weight)

        fused = (
            reciprocal_rank_fusion(
                ranked_lists,
                top_k=self.config.fused_top_k_per_claim,
                weights=weights,
            )
            if ranked_lists
            else []
        )
        scanned = self._predicate_document_scan(claim, predicates, doc_scope)
        navigated = self._structure_navigation_candidates(question, claim, predicates, doc_scope)
        return _dedupe_results([*navigated, *scanned, *fused])

    def _structure_navigation_candidates(
        self,
        question: Question,
        claim: ClaimTarget,
        predicates: list[str],
        doc_scope: list[str],
    ) -> list[RetrievalResult]:
        """先定位自然页/章节，再补充局部证据；该路径永不删除全局 BM25 结果。"""
        if self.structure_navigator is None:
            return []
        entity_terms = [
            term
            for term in extract_query_entities(f"{question.question} {claim.option_text}")
            if term not in extract_candidate_values(claim)
        ]
        query = " ".join(dict.fromkeys([*predicates, *entity_terms[:8]]))
        hits = self.structure_navigator.search(
            query,
            doc_ids=doc_scope,
            top_k_per_doc=self.config.navigation_nodes_per_doc,
        )
        allowed_types = set(self.config.search_chunk_types)
        per_doc: dict[str, list[tuple[float, NavigationHit, Chunk]]] = {}
        values = extract_candidate_values(claim)
        for hit, chunk in self.structure_navigator.expand_chunks(
            hits,
            page_radius=self.config.navigation_page_radius,
            allowed_chunk_types=allowed_types,
        ):
            text = _compact(chunk.text)
            predicate_hits = sum(1 for term in predicates if _compact(term) in text)
            entity_hits = sum(1 for term in entity_terms[:8] if _compact(term) in text)
            value_hits = sum(1 for value in values if _compact(value) in text)
            if predicate_hits == 0 and entity_hits == 0:
                continue
            score = (
                1.8 * predicate_hits
                + 0.7 * entity_hits
                + 0.35 * value_hits
                + 0.5 / max(1, hit.rank_in_doc)
            )
            per_doc.setdefault(chunk.doc_id, []).append((score, hit, chunk))

        output: list[RetrievalResult] = []
        for doc_id in doc_scope:
            ranked = sorted(
                per_doc.get(doc_id, []),
                key=lambda item: (-item[0], item[2].chunk_id),
            )
            for score, hit, chunk in ranked[: self.config.navigation_candidates_per_doc]:
                result = self.index.result_from_chunk(
                    chunk,
                    score=0.4 + min(15.0, score) * 0.03,
                    source=f"{self.config.strategy_name}:{claim.option_key}:structure_navigation",
                    query=query,
                )
                result.metadata["navigation_node_id"] = hit.node_id
                result.metadata["navigation_page"] = hit.page
                result.metadata["navigation_section"] = hit.section
                result.metadata["navigation_rank_in_doc"] = hit.rank_in_doc
                output.append(result)
        return output

    def _predicate_document_scan(
        self,
        claim: ClaimTarget,
        predicates: list[str],
        doc_scope: list[str],
    ) -> list[RetrievalResult]:
        """遍历已限定文档的原子子块，按谓词命中补回 BM25 未排序到前列的真实值。"""
        values = extract_candidate_values(claim)
        allowed_types = set(self.config.search_chunk_types)
        output: list[RetrievalResult] = []
        for doc_id in doc_scope:
            scored: list[tuple[float, Chunk]] = []
            for chunk in self.index.doc_chunks_ordered.get(doc_id, []):
                if str(chunk.metadata.get("chunk_type", "text")) not in allowed_types:
                    continue
                text = _compact(chunk.text)
                predicate_hits = sum(1 for term in predicates if _compact(term) in text)
                if predicate_hits == 0:
                    continue
                value_hits = sum(1 for value in values if _compact(value) in text)
                structure_bonus = (
                    1.0
                    if chunk.metadata.get("chunk_type")
                    in {"table_row", "financial_metric_row", "layout_table_row"}
                    else 0.0
                )
                scored.append((predicate_hits * 2.0 + value_hits * 0.4 + structure_bonus, chunk))
            scored.sort(key=lambda row: row[0], reverse=True)
            for score, chunk in scored[: self.config.predicate_scan_per_doc]:
                output.append(
                    self.index.result_from_chunk(
                        chunk,
                        score=0.3 + min(12.0, score) * 0.02,
                        source=f"{self.config.strategy_name}:{claim.option_key}:predicate_scan",
                        query=" ".join(predicates),
                    )
                )
        return output

    def _restore_parent_excerpt(self, claim: ClaimTarget, item: RetrievalResult) -> RetrievalResult:
        """仅对代词或例外条件不完整的短子块恢复局部父文，不回填整个 1800 字父块。"""
        text = item.evidence_text or ""
        needs_parent = (
            len(text) < 120
            and any(marker in text for marker in ("上述", "其中", "本条", "该项", "但", "除外"))
        ) or (claim.claim_type == "clause_consequence" and not any(marker in text for marker in ("但", "除外", "不得", "应当")))
        if not needs_parent:
            return item
        parent = self.index.get_parent_chunk(item.chunk_id)
        if parent is None or not parent.text:
            return item
        excerpt = _parent_window(parent.text, text, max_chars=720)
        if len(excerpt) <= len(text):
            return item
        restored = RetrievalResult.from_dict(item.to_dict())
        restored.evidence_text = excerpt
        restored.source = f"{item.source}:parent_excerpt"
        restored.metadata["restored_parent_chunk_id"] = parent.chunk_id
        return restored


def build_precise_judge_messages(question: Question, context: str) -> list[dict[str, str]]:
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是金融文档选项核验器，只能依据给定原文证据。每个选项先确定需要核验的谓词，"
                "再从同谓词证据读取真实主体、数值、年份、单位、条件和例外。证据角色 support/counter "
                "只是检索提示，不代表结论。候选值未出现不能单独证明错误；必须找到同谓词真实值或明确否定。"
                "证据头的 title 是文档对应产品/公司，任何事实只能用于匹配该 title 的选项主体，禁止把其他"
                "产品的相似条款交叉套用。复合陈述任一子句错误则整项错误；比较题必须核对每份文档；"
                "checks 中的 truth 必须表示“该选项是否应按题干要求被勾选”，不是括号内解释孤立地是否属实。"
                "例如题干问‘哪些产品可以赔付’，写着‘某产品不赔’的选项即使括号理由属实也必须判 false；"
                "题干问‘哪些说法正确’时才判断整句事实。最后严格按题干选择正确项或错误项，"
                "uncertain 不得当作 true，answer 必须与 checks 及 selection_rule 一致。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题型：{question.answer_format}\n题干：{question.question}\n选项：\n{options}\n\n"
                f"按选项分组的原文证据：\n{context}\n\n"
                "返回单个紧凑 JSON，不要 Markdown："
                '{"checks":{"A":{"truth":"true|false|uncertain","evidence":["A-E1"],"reason":"一句话"}},'
                '"selection_rule":"correct|incorrect","answer":"A或排序后的多选字母","confidence":0.0}'
            ),
        },
    ]


def build_precise_audit_messages(question: Question, context: str, first_response: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "复核初审，只检查否定题、复合断言、年份单位、跨文档比较和例外条件；无明确反证不要改答案。",
        },
        {
            "role": "user",
            "content": (
                f"题干：{question.question}\n证据：\n{context}\n\n初审：{first_response}\n"
                '返回紧凑 JSON：{"correction":"一句话","answer":"字母组合","confidence":0.0}'
            ),
        },
    ]


def _merge_claim_evidence(
    selected_by_claim: list[tuple[ClaimTarget, list[RetrievalResult], dict]],
    max_chars: int,
) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[tuple[str, str]] = set()
    used = 0
    max_items = max((len(items) for _, items, _ in selected_by_claim), default=0)
    for index in range(max_items):
        for claim, items, _ in selected_by_claim:
            if index >= len(items):
                continue
            item = items[index]
            key = (claim.option_key, item.chunk_id)
            length = len(item.evidence_text or "")
            if key in seen or (output and used + length > max_chars):
                continue
            output.append(item)
            seen.add(key)
            used += length
    return output


def _format_grouped_context(
    question: Question,
    evidence: list[RetrievalResult],
    max_chars: int,
) -> tuple[str, dict[str, str]]:
    counters = {key: 0 for key in question.options}
    blocks: list[str] = []
    evidence_map: dict[str, str] = {}
    used = 0
    for item in evidence:
        option_key = str(item.metadata.get("option_key", "?"))
        counters[option_key] = counters.get(option_key, 0) + 1
        evidence_id = f"{option_key}-E{counters[option_key]}"
        role = item.metadata.get("verification_role", "ground_truth")
        block = (
            f"[{evidence_id}][{role}] doc={item.doc_id} page={item.metadata.get('page')} "
            f"title={document_label(question.domain, item.doc_id, item.metadata.get('title', ''))} "
            f"section={item.metadata.get('section', '')}\n{item.evidence_text.strip()}"
        )
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        evidence_map[evidence_id] = item.chunk_id
        used += len(block)
    return "\n\n".join(blocks), evidence_map


def _parent_window(parent_text: str, child_text: str, max_chars: int) -> str:
    parent_text = str(parent_text)
    compact_child = "".join(child_text.split())
    start = parent_text.find(child_text)
    if start < 0 and compact_child:
        anchor = compact_child[: min(24, len(compact_child))]
        compact_parent = "".join(parent_text.split())
        compact_start = compact_parent.find(anchor)
        if compact_start >= 0:
            # 空白归一化后无法可靠反算字符位置，比例映射足以截取邻近上下文。
            start = int(compact_start / max(1, len(compact_parent)) * len(parent_text))
    if start < 0:
        return child_text
    left = max(0, start - max_chars // 3)
    right = min(len(parent_text), left + max_chars)
    return parent_text[left:right]


def _dedupe_results(items: list[RetrievalResult]) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[str] = set()
    for item in items:
        if item.chunk_id in seen:
            continue
        seen.add(item.chunk_id)
        output.append(item)
    return output


def _valid_answer(answer: str, question: Question) -> bool:
    if not answer or any(letter not in question.options for letter in answer):
        return False
    if question.answer_format in {"mcq", "tf"}:
        return len(answer) == 1
    return answer == "".join(sorted(set(answer)))


def _extract_confidence(text: str) -> float:
    obj = extract_json_object(text) or {}
    try:
        return max(0.0, min(1.0, float(obj.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _answer_from_checks(text: str, question: Question) -> str:
    """把逐项 selected truth 程序化组装为答案，避免 checks 与 answer 自相矛盾。"""
    obj = extract_json_object(text) or {}
    checks = obj.get("checks")
    if not isinstance(checks, dict):
        return ""
    verdicts: dict[str, str] = {}
    for option_key in question.options:
        row = checks.get(option_key)
        if not isinstance(row, dict):
            return ""
        verdict = str(row.get("truth", "")).strip().lower()
        if verdict not in {"true", "false", "uncertain"}:
            return ""
        verdicts[option_key] = verdict
    selected = "".join(key for key in sorted(question.options) if verdicts[key] == "true")
    return selected if _valid_answer(selected, question) else ""


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("，", ",").replace("％", "%")
