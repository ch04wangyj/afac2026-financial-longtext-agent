"""脚本 35：从可信提交历史中选择低下行风险的正交探针。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.active_probe import rank_probe_variants
from agent.evaluation.leaderboard_registry import load_verified_leaderboard_runs
from agent.io.jsonl import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="排序排行榜约束下的候选提交。")
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "configs" / "leaderboard_runs.json",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="限定参与硬约束的可信运行名称；默认使用注册表中的全部可信运行。",
    )
    parser.add_argument("--baseline", required=True, help="可信基线运行名称。")
    parser.add_argument(
        "--alternative",
        action="append",
        required=True,
        metavar="QID=ANSWER,ANSWER",
        help="为一道题枚举候选答案；应显式包含要保留的基线答案。",
    )
    parser.add_argument("--max-variants", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected_names = set(args.run) or None
    runs = load_verified_leaderboard_runs(
        args.registry,
        names=selected_names,
    )
    run_by_name = {run.name: run for run in runs}
    if args.baseline not in run_by_name:
        raise KeyError(f"基线未包含在可信运行中: {args.baseline}")
    baseline = run_by_name[args.baseline]

    questions = load_questions(Settings.from_env().questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }
    alternatives = _parse_alternatives(args.alternative)
    ranked = rank_probe_variants(
        runs,
        reference_answers=baseline.answers,
        reference_correct_count=baseline.correct_count,
        alternatives_by_qid=alternatives,
        valid_answers_by_qid=valid_answers,
        max_variants=args.max_variants,
    )
    payload = {
        "purpose": "优先选择最坏结果不低于基线的正交候选",
        "baseline": args.baseline,
        "baseline_correct_count": baseline.correct_count,
        "trusted_runs": [run.name for run in runs],
        "alternatives": {
            qid: list(answers) for qid, answers in alternatives.items()
        },
        "candidate_count": len(ranked),
        "candidates": [candidate.to_dict() for candidate in ranked],
    }
    write_json(args.output, payload)
    print(f"可信运行: {', '.join(run.name for run in runs)}")
    print(f"候选数: {len(ranked)}")
    for index, candidate in enumerate(ranked[:10], start=1):
        changes = ", ".join(
            f"{qid}:{old}->{new}" for qid, old, new in candidate.changes
        )
        outcomes = ",".join(map(str, candidate.possible_correct_counts))
        print(f"{index:02d}. [{outcomes}] {changes}")
    print(f"输出: {args.output.resolve()}")


def _parse_alternatives(values: list[str]) -> dict[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"候选答案格式错误: {value}")
        qid, raw_answers = (part.strip() for part in value.split("=", 1))
        answers = tuple(
            dict.fromkeys(answer.strip() for answer in raw_answers.split(",") if answer.strip())
        )
        if not qid or not answers:
            raise ValueError(f"候选答案格式错误: {value}")
        if qid in output:
            raise ValueError(f"题目 {qid} 重复定义候选答案")
        output[qid] = answers
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


if __name__ == "__main__":
    main()
