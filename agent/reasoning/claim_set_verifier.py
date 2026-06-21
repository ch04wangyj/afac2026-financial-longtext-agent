"""Claim verdict 的证据引用校验与集合级聚合。

模型负责判断语义关系；本模块负责确定性检查证据编号、关键槽位、全称断言
文档覆盖和选项集合冲突，避免单个高置信但无依据的 verdict 直接进入答案。
"""

from __future__ import annotations

import re

from agent.schemas import Question, RetrievalResult


_REF_RE = re.compile(r"\d+")
_UNIVERSAL_HINTS = ("均", "都", "双方", "两份", "两家", "各自", "所有", "分别")
_COMPOUND_HINTS = ("并且", "同时", "以及", "且", "；", ";")
_NUMERIC_HINTS = ("高于", "低于", "超过", "不超过", "同比", "环比", "比例", "占比", "%", "％", "倍")
_CRITICAL_SLOT_PREFIXES = (
    "fact:numeric_value",
    "fact:comparison_endpoints",
    "fact:consequence",
)


def calibrate_claim_verdict(
    *,
    relation: str,
    confidence: float,
    support_evidence: list[str],
    refute_evidence: list[str],
    sufficiency: dict,
    selection_report: dict,
    evidence: list[RetrievalResult],
    doc_scope: list[str],
    option_text: str,
    require_valid_citations: bool = True,
) -> dict:
    """把局部模型 verdict 校准成可用于答案组装的关系。"""
    relation = relation if relation in {"support", "refute", "insufficient"} else "insufficient"
    valid_support = validate_evidence_refs(support_evidence, len(evidence))
    valid_refute = validate_evidence_refs(refute_evidence, len(evidence))
    tags: list[str] = []

    if support_evidence and len(valid_support) < len(support_evidence):
        tags.append("invalid_support_citation")
    if refute_evidence and len(valid_refute) < len(refute_evidence):
        tags.append("invalid_refute_citation")
    if valid_support and valid_refute:
        tags.append("support_refute_conflict")
        relation = "insufficient"
        confidence = min(confidence, 0.45)

    relation_refs = valid_support if relation == "support" else valid_refute if relation == "refute" else []
    if require_valid_citations and relation in {"support", "refute"} and not relation_refs:
        tags.append("missing_relation_citation")
        relation = "insufficient"
        confidence = min(confidence, 0.35)

    missing_slots = list(selection_report.get("missing_slots") or [])
    critical_missing = [slot for slot in missing_slots if slot.startswith(_CRITICAL_SLOT_PREFIXES)]
    if relation == "support" and critical_missing:
        tags.append("critical_evidence_slot_missing")
        relation = "insufficient"
        confidence = min(confidence, 0.40)

    if relation == "support" and _is_universal(option_text) and len(doc_scope) > 1:
        referenced_docs = _referenced_docs(valid_support, evidence)
        missing_docs = [doc_id for doc_id in doc_scope if doc_id not in referenced_docs]
        if missing_docs:
            tags.append("universal_claim_missing_doc_support")
            relation = "insufficient"
            confidence = min(confidence, 0.40)
        else:
            missing_docs = []
    else:
        missing_docs = []

    if not sufficiency.get("sufficient", False):
        tags.extend(str(tag) for tag in sufficiency.get("failure_tags") or [])
        if relation == "support" and not valid_support:
            relation = "insufficient"
            confidence = min(confidence, 0.40)

    return {
        "relation": relation,
        "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "support_evidence": [f"[{index}]" for index in valid_support],
        "refute_evidence": [f"[{index}]" for index in valid_refute],
        "support_chunk_ids": [evidence[index - 1].chunk_id for index in valid_support],
        "refute_chunk_ids": [evidence[index - 1].chunk_id for index in valid_refute],
        "calibration_tags": list(dict.fromkeys(tags)),
        "missing_slots": missing_slots,
        "missing_universal_doc_ids": missing_docs,
    }


def validate_evidence_refs(refs: list[str], evidence_count: int) -> list[int]:
    """抽取并去重 1-based 证据编号，丢弃越界引用。"""
    valid: list[int] = []
    for ref in refs or []:
        match = _REF_RE.search(str(ref))
        if not match:
            continue
        index = int(match.group(0))
        if 1 <= index <= evidence_count and index not in valid:
            valid.append(index)
    return valid


def aggregate_claim_relations(question: Question, verdicts: dict[str, dict]) -> tuple[str, dict]:
    """按题型组装候选答案，并返回集合级冲突信号。"""
    supported = [key for key in sorted(verdicts) if verdicts[key].get("relation") == "support"]
    insufficient = [key for key in sorted(verdicts) if verdicts[key].get("relation") == "insufficient"]
    refuted = [key for key in sorted(verdicts) if verdicts[key].get("relation") == "refute"]
    if question.answer_format == "multi":
        answer = "".join(supported)
    else:
        ranked = sorted(
            supported,
            key=lambda key: (-float(verdicts[key].get("confidence", 0.0) or 0.0), key),
        )
        if not ranked:
            ranked = sorted(
                insufficient,
                key=lambda key: (-float(verdicts[key].get("confidence", 0.0) or 0.0), key),
            )
        answer = ranked[0] if ranked else ""
    report = {
        "supported_options": supported,
        "refuted_options": refuted,
        "insufficient_options": insufficient,
        "multiple_supported_for_single": question.answer_format != "multi" and len(supported) > 1,
        "empty_multi_support": question.answer_format == "multi" and not supported,
    }
    return answer, report


def should_run_claim_set_verification(question: Question, verdicts: dict[str, dict]) -> bool:
    """多选或高风险断言触发一次集合级 exact-match 复核。"""
    if question.answer_format == "multi":
        return True
    if any(value.get("relation") == "insufficient" for value in verdicts.values()):
        return True
    if sum(value.get("relation") == "support" for value in verdicts.values()) != 1:
        return True
    text = " ".join(question.options.values())
    return any(hint in text for hint in (*_UNIVERSAL_HINTS, *_COMPOUND_HINTS, *_NUMERIC_HINTS))


def _referenced_docs(refs: list[int], evidence: list[RetrievalResult]) -> set[str]:
    return {evidence[index - 1].doc_id for index in refs if 1 <= index <= len(evidence)}


def _is_universal(text: str) -> bool:
    return any(hint in (text or "") for hint in _UNIVERSAL_HINTS)
