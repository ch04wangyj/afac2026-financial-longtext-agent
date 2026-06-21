"""面向长文档问答的证据集合选择。

该模块不依赖 embedding。它在稀疏检索与结构化重排之后，用确定性的
覆盖收益选择证据，目标是在固定字符预算内同时覆盖文档、实体、数值、日期
和条款后果，避免若干高分近重复 chunk 占满上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.retrieve.targets import RetrievalTarget
from agent.schemas import RetrievalResult


_CONSEQUENCE_HINTS = ("处罚", "罚款", "扣减", "减分", "责令", "不得", "应当", "期限", "赔付", "免责")
_METRIC_HINTS = ("营业收入", "净利润", "现金流", "分红", "每股", "资产负债率", "保费", "赔付")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d[\d,，]*(?:\.\d+)?\s*(?:%|％|元|千元|万元|亿元|万|亿|倍)?")


@dataclass(frozen=True)
class EvidenceSelectionReport:
    """记录选择后的覆盖情况，供审计和低置信路由使用。"""

    required_slots: list[str]
    covered_slots: list[str]
    missing_slots: list[str]
    selected_chunk_ids: list[str]
    selected_doc_ids: list[str]
    used_chars: int

    @property
    def coverage_ratio(self) -> float:
        if not self.required_slots:
            return 1.0
        return len(self.covered_slots) / len(self.required_slots)

    def to_dict(self) -> dict:
        return {
            "required_slots": list(self.required_slots),
            "covered_slots": list(self.covered_slots),
            "missing_slots": list(self.missing_slots),
            "coverage_ratio": round(self.coverage_ratio, 6),
            "selected_chunk_ids": list(self.selected_chunk_ids),
            "selected_doc_ids": list(self.selected_doc_ids),
            "used_chars": self.used_chars,
        }


def select_evidence_set(
    target: RetrievalTarget,
    candidates: list[RetrievalResult],
    *,
    top_k: int,
    max_chars: int,
    pinned_chunk_ids: set[str] | None = None,
) -> tuple[list[RetrievalResult], EvidenceSelectionReport]:
    """按边际覆盖收益选择证据，并优先保留已被 verdict 引用的 chunk。"""
    if not candidates or top_k <= 0:
        report = EvidenceSelectionReport(
            required_slots=_required_slots(target),
            covered_slots=[],
            missing_slots=_required_slots(target),
            selected_chunk_ids=[],
            selected_doc_ids=[],
            used_chars=0,
        )
        return [], report

    required = _required_slots(target)
    slot_map = {item.chunk_id: _covered_slots(target, item) for item in candidates}
    max_score = max(abs(float(item.score)) for item in candidates) or 1.0
    selected: list[RetrievalResult] = []
    selected_ids: set[str] = set()
    selected_docs: set[str] = set()
    selected_texts: list[str] = []
    covered: set[str] = set()
    used_chars = 0

    # 集合级复核必须能看到局部 verdict 实际引用的原文。固定证据仍受数量和
    # 字符预算约束，按候选原排序保留，随后再用覆盖收益填充剩余位置。
    pinned = set(pinned_chunk_ids or ())
    for item in candidates:
        if item.chunk_id not in pinned or item.chunk_id in selected_ids or len(selected) >= top_k:
            continue
        text = " ".join((item.evidence_text or "").split())
        if not text or (selected and used_chars + len(text) > max_chars):
            continue
        selected.append(item)
        selected_ids.add(item.chunk_id)
        selected_docs.add(item.doc_id)
        selected_texts.append(text)
        covered.update(slot_map[item.chunk_id])
        used_chars += len(text)

    while len(selected) < top_k:
        best: RetrievalResult | None = None
        best_utility = float("-inf")
        for item in candidates:
            if item.chunk_id in selected_ids:
                continue
            text = " ".join((item.evidence_text or "").split())
            if not text:
                continue
            if selected and used_chars + len(text) > max_chars:
                continue

            new_slots = slot_map[item.chunk_id] - covered
            required_gain = len(new_slots & set(required))
            optional_gain = len(new_slots - set(required))
            normalized_score = float(item.score) / max_score
            doc_diversity = 0.8 if item.doc_id not in selected_docs else 0.0
            novelty = _text_novelty(text, selected_texts)
            structure_bonus = _structure_bonus(target, item)
            length_penalty = min(0.45, len(text) / max(1, max_chars) * 0.45)
            utility = (
                required_gain * 2.2
                + optional_gain * 0.25
                + normalized_score * 0.55
                + doc_diversity
                + novelty * 0.45
                + structure_bonus
                - length_penalty
            )
            if utility > best_utility:
                best = item
                best_utility = utility

        if best is None:
            break
        text = " ".join((best.evidence_text or "").split())
        selected.append(best)
        selected_ids.add(best.chunk_id)
        selected_docs.add(best.doc_id)
        selected_texts.append(text)
        covered.update(slot_map[best.chunk_id])
        used_chars += len(text)

    # 如果预算允许但贪心结果为空，保留最高分证据，避免下游收到空上下文。
    if not selected:
        best = max(candidates, key=lambda item: float(item.score))
        selected = [best]
        selected_ids = {best.chunk_id}
        selected_docs = {best.doc_id}
        covered.update(slot_map[best.chunk_id])
        used_chars = len(" ".join((best.evidence_text or "").split()))

    if target.evidence_intent == "comparison":
        selected_values = {
            value
            for item in selected
            for value in _non_year_numeric_mentions(item.evidence_text or "")
        }
        if len(selected_values) >= 2:
            covered.add("fact:comparison_endpoints")

    covered_required = [slot for slot in required if slot in covered]
    report = EvidenceSelectionReport(
        required_slots=required,
        covered_slots=covered_required,
        missing_slots=[slot for slot in required if slot not in covered],
        selected_chunk_ids=[item.chunk_id for item in selected],
        selected_doc_ids=list(dict.fromkeys(item.doc_id for item in selected)),
        used_chars=used_chars,
    )
    return selected, report


def _required_slots(target: RetrievalTarget) -> list[str]:
    slots: list[str] = []
    slots.extend(f"doc:{doc_id}" for doc_id in target.doc_scope)
    slots.extend(f"term:{term}" for term in target.must_terms[:8] if term)
    slots.extend(f"number:{item}" for item in target.numbers[:4] if item)
    slots.extend(f"date:{item}" for item in target.dates[:4] if item)
    if target.evidence_intent in {"number", "comparison"}:
        slots.append("fact:numeric_value")
    if target.evidence_intent == "comparison":
        slots.append("fact:comparison_endpoints")
    if any(hint in target.question for hint in _CONSEQUENCE_HINTS):
        slots.append("fact:consequence")
    return _dedupe(slots)


def _covered_slots(target: RetrievalTarget, item: RetrievalResult) -> set[str]:
    text = " ".join((item.evidence_text or "").split())
    slots = {f"doc:{item.doc_id}"}
    for term in target.must_terms[:8]:
        if term and term in text:
            slots.add(f"term:{term}")
    for number in target.numbers[:4]:
        if number and number in text:
            slots.add(f"number:{number}")
    for date in target.dates[:4]:
        if date and date in text:
            slots.add(f"date:{date}")

    non_year_values = _non_year_numeric_mentions(text)
    if non_year_values:
        slots.add("fact:numeric_value")
    if target.evidence_intent == "comparison" and len(set(non_year_values)) >= 2:
        slots.add("fact:comparison_endpoints")
    if any(hint in text for hint in _CONSEQUENCE_HINTS):
        slots.add("fact:consequence")
    if any(hint in text for hint in _METRIC_HINTS):
        slots.add("fact:metric_block")
    if item.metadata.get("chunk_type") in {"table", "figure"} or item.metadata.get("tables"):
        slots.add("fact:structured_block")
    return slots


def _structure_bonus(target: RetrievalTarget, item: RetrievalResult) -> float:
    text = item.evidence_text or ""
    bonus = 0.0
    if item.metadata.get("clause_id"):
        bonus += 0.20
    if target.evidence_intent in {"number", "comparison"} and (
        item.metadata.get("chunk_type") in {"table", "figure"} or any(hint in text for hint in _METRIC_HINTS)
    ):
        bonus += 0.30
    if any(hint in text for hint in _CONSEQUENCE_HINTS):
        bonus += 0.20
    alias_hits = list(item.metadata.get("claim_metric_alias_hits", []))
    if alias_hits:
        bonus += min(1.20, len(alias_hits) * 0.60)
    return bonus


def _text_novelty(text: str, selected_texts: list[str]) -> float:
    if not selected_texts:
        return 1.0
    grams = _char_ngrams(text)
    max_similarity = 0.0
    for other in selected_texts:
        other_grams = _char_ngrams(other)
        union = grams | other_grams
        similarity = len(grams & other_grams) / max(1, len(union))
        max_similarity = max(max_similarity, similarity)
    return 1.0 - max_similarity


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _is_plain_year(value: str) -> bool:
    compact = re.sub(r"\s+", "", value).rstrip("年")
    return bool(re.fullmatch(r"(?:19|20)\d{2}", compact))


def _non_year_numeric_mentions(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in _NUMBER_RE.finditer(text)
        if not _is_plain_year(match.group(0).strip())
    ]


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))
