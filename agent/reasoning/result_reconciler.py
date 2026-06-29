"""V3 保守结果融合：弱证据回退 V2，强变化再做一次独立审计。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object, parse_answer
from agent.schemas import AnswerResult, Question, TokenUsage


WEAK_EVIDENCE_TERMS = (
    "uncertain",
    "无法判断",
    "证据不足",
    "未提供",
    "未找到",
    "信息不足",
    "无相关证据",
    "缺乏",
)


@dataclass(frozen=True)
class ReconcileConfig:
    """只有独立审计高置信复现 V3 答案时才接受变化。"""

    confidence_threshold: float = 0.8
    max_context_chars: int = 12_000
    max_tokens: int = 1_024
    enable_thinking: bool = False


class ResultReconciler:
    """在低 Token V3 结果与 V2 基线之间保守仲裁。"""

    def __init__(self, llm: QwenClient, config: ReconcileConfig | None = None) -> None:
        self.llm = llm
        self.config = config or ReconcileConfig()

    def reconcile(self, question: Question, current: AnswerResult, baseline: AnswerResult) -> AnswerResult:
        if current.answer == baseline.answer:
            return _copy_with_decision(current, current.answer, "unchanged", baseline.answer)
        if is_weak_result(current, question):
            return _copy_with_decision(current, baseline.answer, "weak_evidence_fallback", baseline.answer)

        context = _format_context(current, self.config.max_context_chars)
        response = self.llm.chat(
            build_reconcile_messages(question, current, baseline, context),
            temperature=0.0,
            max_tokens=self.config.max_tokens,
            enable_thinking=self.config.enable_thinking,
        )
        audit_answer = parse_answer(response.text, question.answer_format)
        confidence = _extract_confidence(response.text)
        accepted = (
            audit_answer == current.answer
            and _valid_answer(audit_answer, question)
            and confidence >= self.config.confidence_threshold
        )
        final_answer = current.answer if accepted else baseline.answer
        decision = "audit_confirmed_v3" if accepted else "audit_fallback_baseline"
        result = _copy_with_decision(current, final_answer, decision, baseline.answer)
        result.token_usage.add(response.usage)
        result.metadata["reconcile_audit_response"] = response.text
        result.metadata["reconcile_audit_answer"] = audit_answer
        result.metadata["reconcile_audit_confidence"] = confidence
        return result


def is_weak_result(result: AnswerResult, question: Question) -> bool:
    """识别 uncertain、证据不足、非法 JSON 和模型空答案等不应覆盖基线的结果。"""
    raw = result.raw_response or ""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return True
    checks = payload.get("checks")
    if not isinstance(checks, dict) or len(checks) < len(question.options):
        return True
    if not str(payload.get("answer", "")).strip():
        return True
    if any(term in raw for term in WEAK_EVIDENCE_TERMS):
        return True
    return any(
        isinstance(check, dict) and str(check.get("truth", "")).lower() == "uncertain"
        for check in checks.values()
    )


def build_reconcile_messages(
    question: Question,
    current: AnswerResult,
    baseline: AnswerResult,
    context: str,
) -> list[dict[str, str]]:
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是独立金融问答审计员。候选答案来自不同检索管线，只能作为待核验假设，不能视为标签。"
                "逐项依据原文核对主体、指标、年份、单位、否定词、例外条件和跨文档端点。"
                "若证据不足以推翻基线，返回基线；只有原文明确支持时才确认新答案。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题型：{question.answer_format}\n题干：{question.question}\n选项：\n{options}\n\n"
                f"V2基线候选：{baseline.answer}\nV3候选：{current.answer}\nV3初审：{current.raw_response}\n\n"
                f"原文证据：\n{context}\n\n"
                '返回紧凑 JSON：{"answer":"排序后的字母组合","confidence":0.0,"reason":"一句话说明关键反证或支持"}'
            ),
        },
    ]


def _format_context(result: AnswerResult, max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for index, item in enumerate(result.evidence, start=1):
        option = item.metadata.get("option_key", "?")
        role = item.metadata.get("verification_role", "ground_truth")
        block = (
            f"[E{index:02d}][option={option}][role={role}] doc={item.doc_id} "
            f"page={item.metadata.get('page')}\n{item.evidence_text.strip()}"
        )
        if blocks and used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _copy_with_decision(
    current: AnswerResult,
    answer: str,
    decision: str,
    baseline_answer: str,
) -> AnswerResult:
    result = AnswerResult.from_dict(current.to_dict())
    result.answer = answer
    result.metadata["strategy"] = "v3_conservative_reconciled"
    result.metadata["v3_initial_answer"] = current.answer
    result.metadata["baseline_answer"] = baseline_answer
    result.metadata["reconcile_decision"] = decision
    return result


def _extract_confidence(text: str) -> float:
    payload = extract_json_object(text) or {}
    try:
        return max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return 0.0


def _valid_answer(answer: str, question: Question) -> bool:
    if not answer or any(letter not in question.options for letter in answer):
        return False
    if question.answer_format in {"mcq", "tf"}:
        return len(answer) == 1
    return answer == "".join(sorted(set(answer)))
