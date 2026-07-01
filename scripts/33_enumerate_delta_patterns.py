"""脚本 33：枚举一次官网提交中增益、回归和双错题的可行组合。"""

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
    parser = argparse.ArgumentParser(
        description="枚举候选提交的可行增益/回归/双错模式。"
    )
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
    if not -change_count <= args.score_delta <= change_count:
        raise ValueError("净分差超出本次变化可能产生的范围")

    # 一道变化题可能出现三种结果：新答案命中、旧答案命中，或两者都未命中。
    # “双错”状态必须显式排除旧、新答案，不能把它误当成零信息。
    contribution = {"gain": 1, "loss": -1, "neutral": 0}
    feasible_patterns: list[dict[str, tuple[str, ...]]] = []
    qids = tuple(changes)
    for states in itertools.product(contribution, repeat=change_count):
        if sum(contribution[state] for state in states) != args.score_delta:
            continue
        state_by_qid = dict(zip(qids, states))
        assignment = {
            qid: changes[qid][1 if state == "gain" else 0]
            for qid, state in state_by_qid.items()
            if state != "neutral"
        }
        forbidden = {
            qid: set(changes[qid])
            for qid, state in state_by_qid.items()
            if state == "neutral"
        }
        if is_partial_assignment_feasible(
            runs,
            valid_answers_by_qid=valid_answers,
            partial_assignment=assignment,
            forbidden_answers_by_qid=forbidden,
        ):
            feasible_patterns.append(
                {
                    state: tuple(
                        qid
                        for qid in qids
                        if state_by_qid[qid] == state
                    )
                    for state in contribution
                }
            )

    print(f"变化题数: {change_count}")
    print(f"净分差: {args.score_delta:+d}")
    print(f"可行模式: {len(feasible_patterns)}")
    for index, pattern in enumerate(feasible_patterns, start=1):
        print(
            f"{index:02d}. 增益={','.join(pattern['gain']) or '-'} | "
            f"回归={','.join(pattern['loss']) or '-'} | "
            f"双错={','.join(pattern['neutral']) or '-'}"
        )


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
