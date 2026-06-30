"""脚本 29：根据多次官网正确题数反推逐题答案可行性。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evaluation.leaderboard_constraints import (
    OTHER_ANSWER,
    LeaderboardRun,
    infer_question_constraints,
)
from agent.config import Settings
from agent.data.questions import load_questions
from agent.io.jsonl import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="执行官网多版本答案约束审计。")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="名称:结果文件:正确题数",
        help="完整 answer_results.jsonl，可重复传入。",
    )
    parser.add_argument("--baseline", required=True, help="作为当前基线的运行名称。")
    parser.add_argument(
        "--fixed-answer",
        action="append",
        default=[],
        metavar="QID=ANSWER",
        help="加入已由原文或分差模式确认的标签条件，可重复传入。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [_parse_run(value) for value in args.run]
    settings = Settings.from_env()
    questions = load_questions(settings.questions_root)
    valid_answers = {
        question.qid: _valid_answers(question.answer_format, question.options)
        for question in questions
    }
    results = infer_question_constraints(
        runs,
        baseline_name=args.baseline,
        valid_answers_by_qid=valid_answers,
        partial_assignment=_parse_fixed_answers(args.fixed_answer),
    )
    forced_wrong = [row for row in results if row.baseline_forced_wrong]
    forced_correct = [row for row in results if row.baseline_forced_correct]
    payload = {
        "purpose": "利用官网总正确题数约束逐题标签可行性",
        "baseline": args.baseline,
        "fixed_answers": _parse_fixed_answers(args.fixed_answer),
        "runs": [
            {
                "name": run.name,
                "correct_count": run.correct_count,
                "result_count": len(run.answers),
            }
            for run in runs
        ],
        "forced_wrong_count": len(forced_wrong),
        "forced_correct_count": len(forced_correct),
        "forced_wrong": [row.to_dict() for row in forced_wrong],
        "questions": [row.to_dict() for row in results],
    }
    write_json(args.output, payload)
    _write_markdown(args.output.with_suffix(".md"), runs, forced_wrong, forced_correct)
    print(f"基线必错题: {len(forced_wrong)}")
    print(f"基线必对题: {len(forced_correct)}")
    print(f"输出: {args.output.resolve()}")


def _parse_run(value: str) -> LeaderboardRun:
    """解析名称、文件和正确题数，路径中允许包含冒号盘符。"""
    try:
        first_colon = value.index(":")
        last_colon = value.rindex(":")
        if first_colon == last_colon:
            raise ValueError
        name = value[:first_colon]
        path_text = value[first_colon + 1 : last_colon]
        count_text = value[last_colon + 1 :]
        correct_count = int(count_text)
    except (ValueError, IndexError) as exc:
        raise argparse.ArgumentTypeError(
            f"运行参数格式错误: {value}; 应为 名称:结果文件:正确题数"
        ) from exc
    path = Path(path_text)
    rows = list(read_jsonl(path))
    answers = {
        str(row["qid"]): str(row["answer"])
        for row in rows
        if row.get("qid") and row.get("qid") != "summary"
    }
    if len(answers) != 100:
        raise ValueError(f"运行 {name} 不是完整 100 题结果: {path}")
    return LeaderboardRun(name=name, answers=answers, correct_count=correct_count)


def _valid_answers(answer_format: str, options: dict[str, str]) -> set[str]:
    """按题型枚举合法答案；多选题允许任意非空字母组合。"""
    keys = sorted(options)
    if answer_format != "multi":
        return set(keys)
    return {
        "".join(group)
        for size in range(1, len(keys) + 1)
        for group in itertools.combinations(keys, size)
    }


def _parse_fixed_answers(values: list[str]) -> dict[str, str]:
    """解析重复的 ``QID=ANSWER`` 参数并拒绝冲突定义。"""
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


def _write_markdown(
    path: Path,
    runs: list[LeaderboardRun],
    forced_wrong,
    forced_correct,
) -> None:
    lines = [
        "# 官网多版本答案约束审计",
        "",
        "## 输入运行",
        "",
        "| 运行 | 正确题数 |",
        "|---|---:|",
    ]
    lines.extend(f"| {run.name} | {run.correct_count} |" for run in runs)
    lines.extend(
        [
            "",
            "## 当前基线必错题",
            "",
            "| QID | 当前答案 | 唯一可行已观测答案 | 全部可行状态 |",
            "|---|---:|---:|---|",
        ]
    )
    for row in forced_wrong:
        feasible = ",".join(
            "其他未观测答案" if answer == OTHER_ANSWER else answer
            for answer in row.feasible_answers
        )
        lines.append(
            f"| {row.qid} | {row.baseline_answer} | "
            f"{row.forced_observed_answer or '-'} | {feasible} |"
        )
    lines.extend(
        [
            "",
            f"当前基线必对题数：`{len(forced_correct)}`。",
            "",
            "“必错”只表示当前答案不满足全部官网总分约束；只有唯一可行已观测答案"
            "才能直接形成答案修正，其余题仍需原文裁决。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
