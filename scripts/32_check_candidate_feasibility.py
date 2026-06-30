"""脚本 32：检查一批答案变化能否同时满足全部官网分数约束。"""

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
    parser = argparse.ArgumentParser(description="执行候选答案联合可行性检查。")
    parser.add_argument(
        "--official-run",
        action="append",
        required=True,
        metavar="名称:结果文件:正确题数",
    )
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    args = parser.parse_args()

    runs = [_parse_run(value) for value in args.official_run]
    baseline = _load_answers(args.baseline_results)
    candidate = _load_answers(args.candidate_results)
    if set(baseline) != set(candidate):
        raise ValueError("基线与候选覆盖的题目集合不一致")
    changes = {
        qid: answer
        for qid, answer in candidate.items()
        if answer != baseline[qid]
    }

    questions = load_questions(Settings.from_env().questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }
    feasible = is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid=valid_answers,
        partial_assignment=changes,
    )
    print(f"变化题数: {len(changes)}")
    print(f"联合可行: {feasible}")
    if not feasible:
        conflicts = _minimal_infeasible_subsets(runs, valid_answers, changes)
        print("最小不可行子集:")
        for conflict in conflicts:
            print("  " + ", ".join(f"{qid}={changes[qid]}" for qid in conflict))


def _minimal_infeasible_subsets(
    runs: list[LeaderboardRun],
    valid_answers: dict[str, set[str]],
    changes: dict[str, str],
) -> list[tuple[str, ...]]:
    """枚举规模较小的最小不可行子集，便于定位互相冲突的改动。"""
    qids = sorted(changes)
    conflicts: list[tuple[str, ...]] = []
    for size in range(1, len(qids) + 1):
        for subset in itertools.combinations(qids, size):
            if any(set(conflict) <= set(subset) for conflict in conflicts):
                continue
            partial = {qid: changes[qid] for qid in subset}
            if not is_partial_assignment_feasible(
                runs,
                valid_answers_by_qid=valid_answers,
                partial_assignment=partial,
            ):
                conflicts.append(subset)
    return conflicts


def _parse_run(value: str) -> LeaderboardRun:
    first = value.index(":")
    last = value.rindex(":")
    if first == last:
        raise ValueError(f"官网运行参数格式错误: {value}")
    name = value[:first]
    path = Path(value[first + 1 : last])
    return LeaderboardRun(
        name=name,
        answers=_load_answers(path),
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
