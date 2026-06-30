"""V6 选项级证据契约。

证据契约不直接猜答案，而是检查当前证据是否覆盖了选项成立所必需的
文档、谓词和数值端点。该报告既用于离线审计，也用于约束 LLM 在证据
不完整时返回 uncertain，避免把相似条款或局部财务口径当成完整证明。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from agent.data.document_aliases import option_doc_scope
from agent.preprocess.chunkers import extract_numbers
from agent.retrieve.claims import build_claim_targets
from agent.retrieve.verification_queries import extract_predicate_terms
from agent.schemas import Question, RetrievalResult


UNIVERSAL_MARKERS = ("均", "全部", "所有", "两份", "各自", "分别", "同时")
COMPARISON_MARKERS = ("高于", "低于", "超过", "少于", "早于", "晚于", "一致", "不一致", "同比")
ABSENCE_MARKERS = (
    "未提及",
    "未披露",
    "未说明",
    "没有",
    "无任何",
    "未出现",
    "均未",
    "不包含",
    "不涵盖",
    "不赔",
    "无免赔额",
)
NEGATION_MARKERS = ("不得", "不能", "不可以", "不属于", "不包括", "除外", "无需", "无权")
COMPOUND_MARKERS = ("且", "并且", "以及", "同时", "分别", "；", ";")
FINANCIAL_SCOPE_MARKERS = (
    "单一客户",
    "单个客户",
    "某一客户",
    "地区信息",
    "分部信息",
    "分产品",
    "分行业",
    "分地区",
    "母公司财务报表",
    "母公司利润表",
    "公司现金流量表",
)
TOTAL_SCOPE_MARKERS = (
    "合计",
    "主要会计数据",
    "合并利润表",
    "合并现金流量表",
    "营业收入(元)",
    "一、营业收入",
)


@dataclass(frozen=True)
class OptionEvidenceContract:
    """单个选项的证据完备性报告。"""

    option_key: str
    required_doc_ids: list[str]
    observed_doc_ids: list[str]
    predicate_terms: list[str]
    predicate_doc_ids: list[str]
    numeric_doc_ids: list[str]
    missing_doc_ids: list[str]
    missing_predicate_doc_ids: list[str]
    missing_numeric_doc_ids: list[str]
    role_counts: dict[str, int]
    risk_tags: list[str]
    coverage_score: float
    selection_ready: bool
    needs_review: bool

    def to_dict(self) -> dict:
        """转换为可直接写入 answer_results.jsonl 的字典。"""
        return asdict(self)


def build_evidence_contracts(
    question: Question,
    evidence: list[RetrievalResult],
) -> dict[str, OptionEvidenceContract]:
    """按选项构建证据契约，并只使用属于该选项的证据。"""
    grouped: dict[str, list[RetrievalResult]] = {}
    for item in evidence:
        option_key = str(item.metadata.get("option_key", ""))
        if option_key:
            grouped.setdefault(option_key, []).append(item)

    contracts: dict[str, OptionEvidenceContract] = {}
    for claim in build_claim_targets(question):
        items = grouped.get(claim.option_key, [])
        required_docs = list(dict.fromkeys(option_doc_scope(question, claim.option_text)))
        predicates = extract_predicate_terms(question, claim)
        observed_docs = _ordered_docs(required_docs, {item.doc_id for item in items})
        predicate_docs = _ordered_docs(
            required_docs,
            {
                item.doc_id
                for item in items
                if _contains_any(item.evidence_text or "", predicates)
            },
        )
        numeric_docs = _ordered_docs(
            required_docs,
            {
                item.doc_id
                for item in items
                if _has_relevant_number(item.evidence_text or "", include_year=claim.claim_type == "date_fact")
            },
        )

        numeric_required = claim.claim_type in {"metric_fact", "comparison", "date_fact"}
        missing_docs = [doc_id for doc_id in required_docs if doc_id not in observed_docs]
        missing_predicates = [doc_id for doc_id in required_docs if doc_id not in predicate_docs]
        missing_numeric = (
            [doc_id for doc_id in required_docs if doc_id not in numeric_docs]
            if numeric_required
            else []
        )
        role_counts = _role_counts(items)
        risk_tags = _risk_tags(
            question,
            claim.option_text,
            items,
            role_counts,
            len(required_docs),
        )

        required_units = len(required_docs) * (3 if numeric_required else 2)
        covered_units = (
            len(observed_docs)
            + len(predicate_docs)
            + (len(numeric_docs) if numeric_required else 0)
        )
        coverage = covered_units / max(1, required_units)
        hard_missing = bool(missing_docs or missing_predicates or missing_numeric)
        review_risk = any(
            tag in risk_tags
            for tag in ("absence_claim", "evidence_conflict", "financial_scope_ambiguity")
        )
        blocks_selection = "financial_scope_ambiguity" in risk_tags
        contracts[claim.option_key] = OptionEvidenceContract(
            option_key=claim.option_key,
            required_doc_ids=required_docs,
            observed_doc_ids=observed_docs,
            predicate_terms=predicates,
            predicate_doc_ids=predicate_docs,
            numeric_doc_ids=numeric_docs,
            missing_doc_ids=missing_docs,
            missing_predicate_doc_ids=missing_predicates,
            missing_numeric_doc_ids=missing_numeric,
            role_counts=role_counts,
            risk_tags=risk_tags,
            coverage_score=round(coverage, 6),
            selection_ready=not hard_missing and not blocks_selection,
            needs_review=hard_missing or review_risk,
        )
    return contracts


def format_evidence_contracts(contracts: dict[str, OptionEvidenceContract]) -> str:
    """生成给裁决模型的紧凑约束文本，不重复原始证据。"""
    lines: list[str] = []
    for key in sorted(contracts):
        contract = contracts[key]
        missing_parts: list[str] = []
        if contract.missing_doc_ids:
            missing_parts.append(f"文档={','.join(contract.missing_doc_ids)}")
        if contract.missing_predicate_doc_ids:
            missing_parts.append(f"谓词={','.join(contract.missing_predicate_doc_ids)}")
        if contract.missing_numeric_doc_ids:
            missing_parts.append(f"数值={','.join(contract.missing_numeric_doc_ids)}")
        missing = "无" if not missing_parts else ";".join(missing_parts)
        risks = ",".join(contract.risk_tags) or "none"
        lines.append(
            f"{key}: coverage={contract.coverage_score:.2f}; "
            f"required_docs={','.join(contract.required_doc_ids)}; "
            f"missing={missing}; risks={risks}; ready={str(contract.selection_ready).lower()}"
        )
    return "\n".join(lines)


def _risk_tags(
    question: Question,
    option_text: str,
    items: list[RetrievalResult],
    role_counts: dict[str, int],
    required_doc_count: int,
) -> list[str]:
    combined = f"{question.question} {option_text}"
    tags: list[str] = []
    if required_doc_count > 1 and any(marker in combined for marker in UNIVERSAL_MARKERS):
        tags.append("universal_scope")
    if any(marker in combined for marker in COMPARISON_MARKERS):
        tags.append("comparison")
    if any(marker in combined for marker in ABSENCE_MARKERS):
        tags.append("absence_claim")
    if any(marker in option_text for marker in NEGATION_MARKERS):
        tags.append("negative_claim")
    if any(marker in option_text for marker in COMPOUND_MARKERS):
        tags.append("compound_claim")
    if role_counts.get("support", 0) and role_counts.get("counter", 0):
        tags.append("evidence_conflict")
    if question.domain == "financial_reports" and _has_financial_scope_ambiguity(option_text, items):
        tags.append("financial_scope_ambiguity")
    return tags


def _has_financial_scope_ambiguity(
    option_text: str,
    items: list[RetrievalResult],
) -> bool:
    """选项问公司整体指标时，局部口径证据必须被显式标记。"""
    if any(marker in option_text for marker in FINANCIAL_SCOPE_MARKERS):
        return False
    texts = [item.evidence_text or "" for item in items]
    has_narrow_scope = any(_contains_any(text, FINANCIAL_SCOPE_MARKERS) for text in texts)
    has_total_scope = any(_contains_any(text, TOTAL_SCOPE_MARKERS) for text in texts)
    return has_narrow_scope and not has_total_scope


def _role_counts(items: list[RetrievalResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        role = str(item.metadata.get("verification_role", "ground_truth"))
        counts[role] = counts.get(role, 0) + 1
    return counts


def _has_relevant_number(text: str, *, include_year: bool) -> bool:
    for value in extract_numbers(text):
        compact = re.sub(r"\s+", "", value).rstrip("年")
        if include_year or not re.fullmatch(r"(?:19|20)\d{2}", compact):
            return True
    return False


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(re.sub(r"\s+", "", term) in compact for term in terms if term)


def _ordered_docs(required_docs: list[str], observed_docs: set[str]) -> list[str]:
    ordered = [doc_id for doc_id in required_docs if doc_id in observed_docs]
    ordered.extend(sorted(observed_docs - set(required_docs)))
    return ordered
