"""V3 确定性证据精排：优先谓词覆盖，并显式保留反证候选。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.preprocess.chunkers import extract_numbers
from agent.retrieve.claims import ClaimTarget
from agent.retrieve.verification_queries import extract_candidate_values
from agent.schemas import RetrievalResult


EXCEPTION_TERMS = ("但", "除外", "仅限", "不得", "应当", "未经", "不包括", "不适用")
STRUCTURED_TYPES = {"table_row", "financial_metric_row", "layout_table_row", "figure"}
NOISE_TERMS = ("目 录", "目录", "风险提示及说明", "释 义")


@dataclass(frozen=True)
class VerificationSelectionReport:
    candidate_count: int
    selected_count: int
    selected_chars: int
    roles: dict[str, int]
    doc_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "selected_chars": self.selected_chars,
            "roles": dict(self.roles),
            "doc_ids": list(self.doc_ids),
        }


def select_verification_evidence(
    claim: ClaimTarget,
    candidates: list[RetrievalResult],
    predicate_terms: list[str],
    *,
    top_k: int = 4,
    max_chars: int = 3600,
) -> tuple[list[RetrievalResult], VerificationSelectionReport]:
    """按文档和 support/counter 角色平衡选择少量高密度证据。"""
    scored = _score_candidates(claim, candidates, predicate_terms)
    selected: list[RetrievalResult] = []
    selected_ids: set[str] = set()
    used_chars = 0

    def add(item: RetrievalResult) -> bool:
        nonlocal used_chars
        length = len(item.evidence_text or "")
        if item.chunk_id in selected_ids or (selected and used_chars + length > max_chars):
            return False
        selected.append(item)
        selected_ids.add(item.chunk_id)
        used_chars += length
        return True

    # 比较题和多文档题先保证每份文档至少有一个真实谓词证据。
    for doc_id in claim.doc_scope:
        item = next((row for row in scored if row.doc_id == doc_id), None)
        if item is not None:
            add(item)
        if len(selected) >= top_k:
            break

    # 再补支持和反证，避免只因候选值不存在就把该选项判为 uncertain。
    for role in ("support", "counter", "ground_truth"):
        item = next((row for row in scored if row.metadata.get("verification_role") == role), None)
        if item is not None:
            add(item)
        if len(selected) >= top_k:
            break

    for item in scored:
        if len(selected) >= top_k:
            break
        add(item)

    roles: dict[str, int] = {}
    for item in selected:
        role = str(item.metadata.get("verification_role", "ground_truth"))
        roles[role] = roles.get(role, 0) + 1
    return selected, VerificationSelectionReport(
        candidate_count=len(candidates),
        selected_count=len(selected),
        selected_chars=used_chars,
        roles=roles,
        doc_ids=list(dict.fromkeys(item.doc_id for item in selected)),
    )


def _score_candidates(
    claim: ClaimTarget,
    candidates: list[RetrievalResult],
    predicate_terms: list[str],
) -> list[RetrievalResult]:
    unique: dict[str, RetrievalResult] = {}
    for item in candidates:
        current = unique.get(item.chunk_id)
        if current is None or float(item.score) > float(current.score):
            unique[item.chunk_id] = item
    max_base = max((abs(float(item.score)) for item in unique.values()), default=1.0) or 1.0
    candidate_values = extract_candidate_values(claim)
    candidate_numbers = {_compact(value) for value in claim.numbers}

    for item in unique.values():
        text = item.evidence_text or ""
        compact = _compact(text)
        predicate_hits = sum(1 for term in predicate_terms if _compact(term) in compact)
        value_hits = sum(1 for value in candidate_values if _compact(value) in compact)
        actual_numbers = {
            _compact(value)
            for value in extract_numbers(text)
            if not re.fullmatch(r"(?:19|20)\d{2}年?", _compact(value))
        }
        alternative_values = actual_numbers - candidate_numbers
        chunk_type = str(item.metadata.get("chunk_type", "text"))
        role = "ground_truth"
        if predicate_hits and value_hits:
            role = "support"
        elif predicate_hits and alternative_values:
            role = "counter"

        score = float(item.score) / max_base
        score += min(4, predicate_hits) * 1.45
        score += min(3, value_hits) * 0.55
        if role == "counter":
            score += 0.85
        if chunk_type in STRUCTURED_TYPES:
            score += 0.75
        if chunk_type == "layout_table_row" and item.metadata.get("table_header"):
            # V4 行同时携带标题、单位和表头，数值语义比普通页面文本更完整。
            score += 0.55
        score += _financial_metric_bonus(claim, item, predicate_terms)
        if any(term in text for term in EXCEPTION_TERMS):
            score += 0.25
        if any(term in text for term in NOISE_TERMS):
            score -= 1.5
        if len(text) > 700:
            score -= min(1.0, (len(text) - 700) / 1000)

        item.metadata["verification_role"] = role
        item.metadata["verification_score"] = round(score, 6)
        item.metadata["predicate_hits"] = predicate_hits
        item.metadata["candidate_value_hits"] = value_hits

    return sorted(
        unique.values(),
        key=lambda item: (
            float(item.metadata.get("verification_score", 0.0)),
            -len(item.evidence_text or ""),
            item.chunk_id,
        ),
        reverse=True,
    )


def _financial_metric_bonus(
    claim: ClaimTarget,
    item: RetrievalResult,
    predicate_terms: list[str],
) -> float:
    """结构化财务行必须匹配当前指标和年份，避免普通叙述中的同名词抢占 Top-K。"""
    row = item.metadata.get("financial_row") or {}
    metric = str(row.get("metric", ""))
    if not metric:
        return 0.0
    normalized_predicates = {_canonical_metric(term) for term in predicate_terms}
    if _canonical_metric(metric) not in normalized_predicates:
        return -2.5
    bonus = 2.4
    cells = list(row.get("cells") or [])
    if len(cells) >= 2 and not row.get("header"):
        # 多值行没有表头时通常是季度列或解析错位，不能用于年度值比较。
        return -1.5
    if len(cells) >= 2:
        bonus += 0.45
    claim_years = {
        re.sub(r"\D", "", value)
        for value in claim.numbers
        if re.fullmatch(r"(?:19|20)\d{2}年?", re.sub(r"\s+", "", value))
    }
    row_years = {str(cell.get("year", "")) for cell in cells if cell.get("year")}
    if claim_years & row_years:
        bonus += 0.45
    return bonus


def _canonical_metric(value: str) -> str:
    compact = _compact(value)
    aliases = (
        (("研发投入强度", "研发投入占比", "研发费用占营业收入比例"), "研发投入占营业收入比例"),
        (("归母净利润", "母公司拥有人应占溢利"), "归属于上市公司股东的净利润"),
        (("经营活动现金流净额", "经营活动现金流量净额"), "经营活动产生的现金流量净额"),
        (("营业总收入", "营业额"), "营业收入"),
    )
    for variants, canonical in aliases:
        if any(_compact(variant) in compact for variant in variants):
            return canonical
    return compact


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).replace("，", ",").replace("％", "%")
