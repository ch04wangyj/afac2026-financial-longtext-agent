"""脚本 09：使用人工核验开发集评估答案完全匹配与关键证据召回。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.answer_devset import AnswerDevCase, evaluate_answer_devset, question_sha1
from agent.io.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer_results.jsonl against a reviewed dev set.")
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--devset", type=Path, default=ROOT / "devsets" / "answer_level_v1.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless all cases are present and correct.")
    args = parser.parse_args()

    cases = [AnswerDevCase.from_dict(row) for row in read_jsonl(args.devset)]
    result_rows = [row for path in args.results for row in read_jsonl(path)]
    settings = Settings.from_env()
    fingerprints = {question.qid: question_sha1(question) for question in load_questions(settings.questions_root)}
    report = evaluate_answer_devset(cases, result_rows, current_question_sha1=fingerprints)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.strict and (
        not report["all_present"]
        or not report["all_question_versions_match"]
        or report["correct"] != report["total"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
