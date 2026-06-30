"""脚本 28：比较多种检索/裁决运行，输出选项级一致翻转候选。"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.option_consensus import (
    audit_option_consensus,
    candidate_answer_from_consensus,
)
from agent.io.jsonl import read_jsonl, write_json
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="生成选项级多运行共识审计。")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs" / "releases" / "v7" / "answer_results.jsonl",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="候选 answer_results.jsonl；可重复传入。",
    )
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings.from_env()
    questions = load_questions(settings.questions_root)
    baseline = _load_results(args.baseline)
    candidate_runs = _load_named_runs(args.candidate)
    rows = audit_option_consensus(
        questions,
        baseline,
        candidate_runs,
        min_runs=args.min_runs,
    )
    unanimous = [row for row in rows if row.unanimous_flip]

    by_qid: dict[str, list] = defaultdict(list)
    for row in unanimous:
        by_qid[row.qid].append(row)
    question_by_qid = {question.qid: question for question in questions}
    baseline_by_qid = {row.qid: row for row in baseline}
    answer_drafts = {
        qid: {
            "baseline_answer": baseline_by_qid[qid].answer,
            "consensus_draft": candidate_answer_from_consensus(
                question_by_qid[qid],
                baseline_by_qid[qid].answer,
                option_rows,
            ),
            "flipped_options": [row.option_key for row in option_rows],
        }
        for qid, option_rows in sorted(by_qid.items())
    }
    payload = {
        "purpose": "候选审计，不自动发布答案",
        "baseline": str(args.baseline.resolve()),
        "candidate_runs": {
            name: len(results) for name, results in candidate_runs.items()
        },
        "min_runs": args.min_runs,
        "unanimous_flip_count": len(unanimous),
        "answer_drafts": answer_drafts,
        "options": [row.to_dict() for row in rows],
    }
    write_json(args.output, payload)
    _write_markdown(args.output.with_suffix(".md"), unanimous, answer_drafts)
    print(f"一致翻转选项: {len(unanimous)}")
    print(f"涉及题目: {len(answer_drafts)}")
    print(f"输出: {args.output.resolve()}")


def _load_named_runs(paths: list[Path]) -> dict[str, list[AnswerResult]]:
    output: dict[str, list[AnswerResult]] = {}
    for path in paths:
        name = path.parent.name
        if name in output:
            raise ValueError(f"候选运行目录名重复: {name}")
        output[name] = _load_results(path)
    return output


def _load_results(path: Path) -> list[AnswerResult]:
    rows = [AnswerResult.from_dict(row) for row in read_jsonl(path)]
    if not rows:
        raise RuntimeError(f"结果文件为空: {path}")
    return rows


def _write_markdown(path: Path, rows: list, answer_drafts: dict) -> None:
    lines = [
        "# 选项级多运行共识审计",
        "",
        "该报告只生成复核候选，不自动修改官网基线。",
        "",
        "| QID | 选项 | 基线选中 | 反向票/运行数 | 草案 | 选项文本 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        draft = answer_drafts[row.qid]["consensus_draft"]
        lines.append(
            f"| {row.qid} | {row.option_key} | {row.baseline_selected} | "
            f"{row.opposite_votes}/{row.observed_runs} | {draft} | {row.option_text} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
