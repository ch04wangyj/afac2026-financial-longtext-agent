"""比赛提交文件生成工具。"""

from __future__ import annotations

import csv
from pathlib import Path

from agent.schemas import AnswerResult, TokenUsage


SUBMISSION_FIELDS = ["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"]


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
