from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.reasoning.answer_parser import extract_json_object


@dataclass(frozen=True)
class OptionVerdict:
    option: str
    verdict: bool | None
    confidence: float = 0.0
    support_evidence: list[str] = field(default_factory=list)
    refute_evidence: list[str] = field(default_factory=list)
    reason: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "support_evidence": list(self.support_evidence),
            "refute_evidence": list(self.refute_evidence),
            "reason": self.reason,
            "raw_response": self.raw_response,
        }



def synthesize_answer(verdicts: dict[str, OptionVerdict], answer_format: str) -> str:
    if answer_format == "multi":
        return "".join(sorted(key for key, verdict in verdicts.items() if verdict.verdict is True))

    true_options = [verdict for verdict in verdicts.values() if verdict.verdict is True]
    if true_options:
        return max(true_options, key=lambda item: item.confidence).option

    known_options = [verdict for verdict in verdicts.values() if verdict.verdict is not None]
    if known_options:
        return max(known_options, key=lambda item: item.confidence).option

    return ""



def parse_option_verdict(text: str, option_key: str) -> OptionVerdict:
    obj = extract_json_object(text) or {}
    relation = str(obj.get("relation", "")).strip().lower()
    if relation == "support":
        verdict = True
    elif relation == "refute":
        verdict = False
    else:
        verdict = None

    try:
        confidence = max(0.0, min(1.0, float(obj.get("confidence", 0.0) or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0

    return OptionVerdict(
        option=option_key,
        verdict=verdict,
        confidence=confidence,
        support_evidence=list(obj.get("support_evidence") or []),
        refute_evidence=list(obj.get("refute_evidence") or []),
        reason=str(obj.get("reason", ""))[:120],
        raw_response=text or "",
    )
