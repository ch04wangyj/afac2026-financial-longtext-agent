"""比赛提交文件生成工具。"""

from __future__ import annotations

import csv
from pathlib import Path

from agent.schemas import AnswerResult, Question, TokenUsage


SUBMISSION_FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


def merge_answer_results(
    base_results: list[AnswerResult],
    override_batches: list[list[AnswerResult]],
) -> list[AnswerResult]:
    """按 qid 覆盖完整结果，并保持底稿中的题目顺序。"""
    if not base_results:
        raise ValueError("base results must not be empty")

    ordered_qids = [result.qid for result in base_results]
    if len(ordered_qids) != len(set(ordered_qids)):
        raise ValueError("base results contain duplicate qids")

    merged = {result.qid: result for result in base_results}
    for batch in override_batches:
        batch_qids = [result.qid for result in batch]
        if len(batch_qids) != len(set(batch_qids)):
            raise ValueError("override results contain duplicate qids")
        unknown = sorted(set(batch_qids) - set(merged))
        if unknown:
            raise ValueError(f"override results contain unknown qids: {unknown}")
        merged.update({result.qid: result for result in batch})
    return [merged[qid] for qid in ordered_qids]


def validate_answer_results(
    results: list[AnswerResult],
    questions: list[Question],
    *,
    require_complete: bool = False,
) -> None:
    """校验题目覆盖、答案格式和逐题 Token，阻止无效 CSV 落盘。"""
    question_by_qid = {question.qid: question for question in questions}
    result_qids = [result.qid for result in results]
    if len(result_qids) != len(set(result_qids)):
        raise ValueError("results contain duplicate qids")

    unknown = sorted(set(result_qids) - set(question_by_qid))
    if unknown:
        raise ValueError(f"results contain unknown qids: {unknown}")
    if require_complete:
        missing = sorted(set(question_by_qid) - set(result_qids))
        if missing:
            raise ValueError(f"submission is missing {len(missing)} qids: {missing}")

    for result in results:
        question = question_by_qid[result.qid]
        answer = result.answer.strip()
        allowed = set(question.options)
        if not answer or any(letter not in allowed for letter in answer):
            raise ValueError(f"invalid answer for {result.qid}: {result.answer!r}")
        if question.answer_format in {"mcq", "tf"} and len(answer) != 1:
            raise ValueError(f"{question.answer_format} answer must be one letter: {result.qid}={answer}")
        if question.answer_format == "multi" and answer != "".join(sorted(set(answer))):
            raise ValueError(f"multi answer must be sorted and unique: {result.qid}={answer}")

        usage = result.token_usage
        if min(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) < 0:
            raise ValueError(f"negative token usage for {result.qid}")
        if usage.total_tokens != usage.prompt_tokens + usage.completion_tokens:
            raise ValueError(f"inconsistent token usage for {result.qid}")


def summarize_usage(results: list[AnswerResult]) -> TokenUsage:
    """汇总所有题目的 Token 用量。"""
    total = TokenUsage()
    for result in results:
        total.add(result.token_usage)
    return total


def write_answer_csv(path: Path, results: list[AnswerResult]) -> None:
    """写出比赛要求的 answer.csv，首行 summary 汇总 Token。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    total = summarize_usage(results)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "qid": "summary",
                "answer": "",
                "prompt_tokens": total.prompt_tokens,
                "completion_tokens": total.completion_tokens,
                "total_tokens": total.total_tokens,
            }
        )
        for result in results:
            usage = result.token_usage
            writer.writerow(
                {
                    "qid": result.qid,
                    "answer": result.answer,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            )
