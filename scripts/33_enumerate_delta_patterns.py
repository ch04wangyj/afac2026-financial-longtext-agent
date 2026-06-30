"""脚本 33：枚举一次官网提交中增益题与回归题的可行组合。"""

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
    is_partial_assignment_feasible,
)
from agent.io.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="枚举候选提交的可行增益/回归模式。")
    parser.add_argument(
        "--official-run",
        action="append",
        required=True,
        metavar="名称:结果文件:正确题数",
    )
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--score-delta", type=int, required=True)
    args = parser.parse_args()

    runs = [_parse_run(value) for value in args.official_run]
    baseline = _load_answers(args.baseline_results)
    candidate = _load_answers(args.candidate_results)
    changes = {
        qid: (baseline[qid], candidate[qid])
        for qid in sorted(baseline)
        if baseline[qid] != candidate[qid]
    }
    questions = load_questions(Settings.from_env().questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }

    change_count = len(changes)
    numerator = change_count + args.score_delta
    if numerator % 2:
        raise ValueError("只考虑旧/新答案时，变化题数与净分差的奇偶性不兼容")
    gain_count = numerator // 2
    if not 0 <= gain_count <= change_count:
        raise ValueError("净分差超出本次变化可能产生的范围")

    feasible_patterns: list[tuple[str, ...]] = []
    qids = tuple(changes)
    for gains in itertools.combinations(qids, gain_count):
        gain_set = set(gains)
        assignment = {
            qid: new_answer if qid in gain_set else old_answer
            for qid, (old_answer, new_answer) in changes.items()
        }
        if is_partial_assignment_feasible(
            runs,
            valid_answers_by_qid=valid_answers,
            partial_assignment=assignment,
        ):
            feasible_patterns.append(gains)

    print(f"变化题数: {change_count}")
    print(f"净分差: {args.score_delta:+d}")
    print(f"假设无第三答案时: {gain_count} 个增益，{change_count - gain_count} 个回归")
    print(f"可行模式: {len(feasible_patterns)}")
    for index, gains in enumerate(feasible_patterns, start=1):
        losses = [qid for qid in qids if qid not in set(gains)]
        print(f"{index:02d}. 增益={','.join(gains)} | 回归={','.join(losses)}")


def _parse_run(value: str) -> LeaderboardRun:
    first = value.index(":")
    last = value.rindex(":")
    if first == last:
        raise ValueError(f"官网运行参数格式错误: {value}")
    return LeaderboardRun(
        name=value[:first],
        answers=_load_answers(Path(value[first + 1 : last])),
        correct_count=int(value[last + 1 :]),
    )


def _load_answers(path: Path) -> dict[str, str]:
    return {
        str(row["qid"]): str(row["answer"])
        for row in read_jsonl(path)
        if row.get("qid") and row.get("qid") != "summary"
    }


def _valid_answers(answer_format: str, options: dict[str, str]) -> set[str]:
    keys = sorted(options)
    if answer_format != "multi":
        return set(keys)
    return {
        "".join(group)
        for size in range(1, len(keys) + 1)
        for group in itertools.combinations(keys, size)
    }


if __name__ == "__main__":
    main()
