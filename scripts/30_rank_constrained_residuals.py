"""脚本 30：在官网总分硬约束内执行多运行加权残差排序。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.leaderboard_constraints import (
    LeaderboardRun,
    infer_weighted_assignment,
)
from agent.io.jsonl import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="生成受官网分数约束的残差候选。")
    parser.add_argument(
        "--official-run",
        action="append",
        required=True,
        metavar="名称:结果文件:正确题数",
    )
    parser.add_argument(
        "--vote-run",
        action="append",
        required=True,
        metavar="名称:结果文件:权重",
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--fixed-answer",
        action="append",
        default=[],
        metavar="QID=ANSWER",
        help="加入已确认的真实标签条件，可重复传入。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    official_runs = [_parse_official_run(value) for value in args.official_run]
    vote_runs = [_parse_vote_run(value) for value in args.vote_run]
    settings = Settings.from_env()
    questions = load_questions(settings.questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }
    weights = _collect_weights(vote_runs)
    assignment = infer_weighted_assignment(
        official_runs,
        baseline_name=args.baseline,
        valid_answers_by_qid=valid_answers,
        answer_weights=weights,
        partial_assignment=_parse_fixed_answers(args.fixed_answer),
    )
    baseline = next(run for run in official_runs if run.name == args.baseline)
    changed = [
        {
            "qid": qid,
            "baseline_answer": baseline.answers[qid],
            "predicted_answer": assignment[qid],
            "answer_weights": weights.get(qid, {}),
            "supporting_runs": [
                name
                for name, answers, _ in vote_runs
                if answers.get(qid) == assignment[qid]
            ],
        }
        for qid in sorted(assignment)
        if assignment[qid] != baseline.answers[qid]
    ]
    payload = {
        "purpose": "候选排序，不自动发布答案",
        "baseline": args.baseline,
        "predicted_correct_count": sum(
            assignment[qid] == baseline.answers[qid] for qid in assignment
        ),
        "changed_count": len(changed),
        "changed": changed,
        "assignment": assignment,
    }
    write_json(args.output, payload)
    _write_markdown(args.output.with_suffix(".md"), changed)
    print(f"相对基线预测错误: {len(changed)}")
    print(f"输出: {args.output.resolve()}")


def _parse_official_run(value: str) -> LeaderboardRun:
    name, path, count = _split_triplet(value)
    answers = _load_answers(path)
    if len(answers) != 100:
        raise ValueError(f"官网运行 {name} 不是完整 100 题")
    return LeaderboardRun(name=name, answers=answers, correct_count=int(count))


def _parse_vote_run(value: str) -> tuple[str, dict[str, str], float]:
    name, path, weight = _split_triplet(value)
    return name, _load_answers(path), float(weight)


def _split_triplet(value: str) -> tuple[str, Path, str]:
    """按首尾冒号切分，兼容 Windows 绝对路径中的盘符冒号。"""
    first = value.index(":")
    last = value.rindex(":")
    if first == last:
        raise ValueError(f"参数格式错误: {value}")
    return value[:first], Path(value[first + 1 : last]), value[last + 1 :]


def _load_answers(path: Path) -> dict[str, str]:
    return {
        str(row["qid"]): str(row["answer"])
        for row in read_jsonl(path)
        if row.get("qid") and row.get("qid") != "summary"
    }


def _collect_weights(
    vote_runs: list[tuple[str, dict[str, str], float]],
) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    for _, answers, run_weight in vote_runs:
        for qid, answer in answers.items():
            by_answer = weights.setdefault(qid, {})
            by_answer[answer] = by_answer.get(answer, 0.0) + run_weight
    return weights


def _parse_fixed_answers(values: list[str]) -> dict[str, str]:
    """解析重复的 ``QID=ANSWER`` 参数。"""
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"固定答案格式错误: {value}")
        qid, answer = (part.strip() for part in value.split("=", 1))
        if not qid or not answer:
            raise ValueError(f"固定答案格式错误: {value}")
        if qid in output and output[qid] != answer:
            raise ValueError(f"题目 {qid} 存在冲突的固定答案")
        output[qid] = answer
    return output


def _valid_answers(answer_format: str, options: dict[str, str]) -> set[str]:
    keys = sorted(options)
    if answer_format != "multi":
        return set(keys)
    return {
        "".join(group)
        for size in range(1, len(keys) + 1)
        for group in itertools.combinations(keys, size)
    }


def _write_markdown(path: Path, changed: list[dict]) -> None:
    lines = [
        "# 官网约束下的多运行残差排序",
        "",
        "该结果是加权候选，不是逐题官方标签，不得直接覆盖提交基线。",
        "",
        "| QID | 基线 | 预测 | 支持运行 | 权重 |",
        "|---|---:|---:|---|---|",
    ]
    for row in changed:
        predicted = row["predicted_answer"]
        score = row["answer_weights"].get(predicted, 0.0)
        lines.append(
            f"| {row['qid']} | {row['baseline_answer']} | {predicted} | "
            f"{','.join(row['supporting_runs']) or '-'} | {score:.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
