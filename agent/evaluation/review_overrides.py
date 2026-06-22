"""对人工逐条核对后的少量答案执行可审计覆盖。"""

from __future__ import annotations

from agent.schemas import AnswerResult


def apply_review_overrides(
    rows: list[AnswerResult],
    overrides: dict[str, dict],
) -> list[AnswerResult]:
    """只修改答案和审计元数据，保留原证据与真实 Token 统计。"""
    by_qid = {row.qid: row for row in rows}
    missing = sorted(set(overrides) - set(by_qid))
    if missing:
        raise KeyError(f"Unknown override qids: {missing}")

    output: list[AnswerResult] = []
    for source in rows:
        result = AnswerResult.from_dict(source.to_dict())
        override = overrides.get(result.qid)
        if override:
            answer = str(override.get("answer", "")).strip()
            if not answer:
                raise ValueError(f"Empty reviewed answer for {result.qid}")
            result.metadata["pre_review_answer"] = result.answer
            result.answer = answer
            result.metadata["final_review"] = {
                "answer": answer,
                "decision": str(override.get("decision", "manual_evidence_review")),
                "reason": str(override.get("reason", "")),
            }
        output.append(result)
    return output
