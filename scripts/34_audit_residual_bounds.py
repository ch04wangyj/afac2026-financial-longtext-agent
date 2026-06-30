"""脚本 34：按领域和历史答案轨迹计算当前基线的残差上下界。"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.leaderboard_constraints import (
    LeaderboardRun,
    infer_correctness_bounds,
)
from agent.io.jsonl import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="计算答案基线的分组正确数上下界。")
    parser.add_argument(
        "--official-run",
        action="append",
        required=True,
        metavar="名称:结果文件:正确题数",
    )
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument(
        "--fixed-answer",
        action="append",
        default=[],
        metavar="QID=ANSWER",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [_parse_run(value) for value in args.official_run]
    baseline = _load_answers(args.baseline_results)
    fixed_answers = _parse_fixed_answers(args.fixed_answer)
    questions = load_questions(Settings.from_env().questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }

    groups: list[dict] = []
    by_domain: dict[str, set[str]] = defaultdict(set)
    for question in questions:
        by_domain[question.domain].add(question.qid)
    for domain, qids in sorted(by_domain.items()):
        bounds = infer_correctness_bounds(
            runs,
            baseline_answers=baseline,
            valid_answers_by_qid=valid_answers,
            partial_assignment=fixed_answers,
            subset_qids=qids,
        )
        groups.append({"kind": "domain", "name": domain, **bounds.to_dict()})

    by_signature: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for qid in sorted(baseline):
        signature = tuple(run.answers[qid] for run in runs)
        by_signature[signature].add(qid)
    for index, (signature, qids) in enumerate(
        sorted(by_signature.items(), key=lambda item: (-len(item[1]), item[0])),
        start=1,
    ):
        bounds = infer_correctness_bounds(
            runs,
            baseline_answers=baseline,
            valid_answers_by_qid=valid_answers,
            partial_assignment=fixed_answers,
            subset_qids=qids,
        )
        groups.append(
            {
                "kind": "history_signature",
                "name": f"signature_{index:02d}",
                "answers_by_run": {
                    run.name: answer for run, answer in zip(runs, signature)
                },
                "qids": sorted(qids),
                **bounds.to_dict(),
            }
        )

    payload = {
        "purpose": "在官网总分与固定标签条件下定位剩余错误分布",
        "baseline_results": str(args.baseline_results),
        "fixed_answers": fixed_answers,
        "groups": groups,
    }
    write_json(args.output, payload)
    _write_markdown(args.output.with_suffix(".md"), groups)
    print(f"分组数: {len(groups)}")
    print(f"输出: {args.output.resolve()}")


def _write_markdown(path: Path, groups: list[dict]) -> None:
    lines = [
        "# 条件化残差正确数区间",
        "",
        "区间只表示与全部官网总分约束相容的范围，不代替逐题原文裁决。",
        "",
        "## 领域",
        "",
        "| 领域 | 题数 | 正确数 | 错误数 |",
        "|---|---:|---:|---:|",
    ]
    for row in groups:
        if row["kind"] != "domain":
            continue
        lines.append(
            f"| {row['name']} | {row['question_count']} | "
            f"{row['min_correct']}-{row['max_correct']} | "
            f"{row['min_wrong']}-{row['max_wrong']} |"
        )
    lines.extend(
        [
            "",
            "## 历史答案轨迹",
            "",
            "| 轨迹 | 题数 | 正确数 | 错误数 | QID |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in groups:
        if row["kind"] != "history_signature":
            continue
        lines.append(
            f"| {row['name']} | {row['question_count']} | "
            f"{row['min_correct']}-{row['max_correct']} | "
            f"{row['min_wrong']}-{row['max_wrong']} | "
            f"{', '.join(row['qids'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _parse_fixed_answers(values: list[str]) -> dict[str, str]:
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


if __name__ == "__main__":
    main()
