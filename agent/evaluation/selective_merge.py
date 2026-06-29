"""把局部候选运行与官网已验证基线做保守、可审计融合。"""

from __future__ import annotations

from agent.schemas import AnswerResult


def merge_candidate_with_baseline(
    baseline_rows: list[AnswerResult],
    candidate_rows: list[AnswerResult],
    reviewed_answers: dict[str, dict],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> list[AnswerResult]:
    """候选覆盖其运行范围；差异题默认回退，只有逐条复核配置可改答案。"""
    baseline = {row.qid: row for row in baseline_rows}
    candidate = {row.qid: row for row in candidate_rows}
    unknown = sorted((set(candidate) | set(reviewed_answers)) - set(baseline))
    if unknown:
        raise KeyError(f"Unknown qids in selective merge: {unknown}")

    output: list[AnswerResult] = []
    for baseline_source in baseline_rows:
        current = candidate.get(baseline_source.qid)
        if current is None:
            output.append(AnswerResult.from_dict(baseline_source.to_dict()))
            continue

        result = AnswerResult.from_dict(current.to_dict())
        candidate_answer = result.answer
        review = reviewed_answers.get(result.qid)
        if review:
            result.answer = str(review.get("answer", "")).strip()
            if not result.answer:
                raise ValueError(f"Empty reviewed answer for {result.qid}")
            decision = str(review.get("decision", "direct_source_review"))
        elif candidate_answer == baseline_source.answer:
            decision = "candidate_unchanged"
        else:
            result.answer = baseline_source.answer
            decision = "candidate_difference_fallback"

        result.metadata["strategy"] = f"{candidate_label}_selective_merge"
        result.metadata["candidate_label"] = candidate_label
        result.metadata["candidate_answer"] = candidate_answer
        result.metadata["baseline_label"] = baseline_label
        result.metadata["baseline_answer"] = baseline_source.answer
        result.metadata["merge_decision"] = decision
        if review:
            result.metadata["source_review"] = {
                "answer": result.answer,
                "reason": str(review.get("reason", "")),
            }
        output.append(result)
    return output
