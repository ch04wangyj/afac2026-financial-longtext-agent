"""脚本 04：根据 answer_results.jsonl 生成提交和审计产物。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.io.jsonl import read_jsonl, write_json, write_jsonl
from agent.io.output_layout import infer_artifact_dir_from_results
from agent.io.submission import (
    merge_answer_results,
    summarize_usage,
    validate_answer_results,
    write_answer_csv,
)
from agent.schemas import AnswerResult


def main() -> None:
    """写出 answer.csv、evidence.json 和 token_usage.json。"""
    parser = argparse.ArgumentParser(description="Create answer.csv, evidence.json, and token_usage.json.")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--override-results",
        type=Path,
        action="append",
        default=[],
        help="按 qid 覆盖底稿结果；可重复传入，后传入者优先。",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="显式指定提交产物目录。")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="要求结果完整覆盖当前 questions/group_a，正式提交时应开启。",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    results_path = args.results or settings.outputs_dir / "answer_results.jsonl"
    if args.override_results and args.output_dir is None:
        raise ValueError("使用 --override-results 时必须指定 --output-dir，避免覆盖结果底稿。")
    artifact_dir = args.output_dir.resolve() if args.output_dir else infer_artifact_dir_from_results(results_path)
    base_results = [AnswerResult.from_dict(row) for row in read_jsonl(results_path)]
    if not base_results:
        raise RuntimeError(f"No results found: {results_path}")
    override_batches = []
    for path in args.override_results:
        batch = [AnswerResult.from_dict(row) for row in read_jsonl(path)]
        if not batch:
            raise RuntimeError(f"No override results found: {path}")
        override_batches.append(batch)
    results = merge_answer_results(base_results, override_batches)
    questions = load_questions(settings.questions_root)
    validate_answer_results(results, questions, require_complete=args.require_complete)

    # 合并后的 JSONL 与 CSV 同目录保存，保证提交答案仍可追溯到证据和 Token。
    write_jsonl(artifact_dir / "answer_results.jsonl", (result.to_dict() for result in results))
    write_answer_csv(artifact_dir / "answer.csv", results)
    write_json(
        artifact_dir / "evidence.json",
        {result.qid: evidence_items_with_qid(result) for result in results},
    )
    usage = summarize_usage(results)
    write_json(artifact_dir / "token_usage.json", usage.to_dict())
    print(f"wrote {len(results)} validated submission rows to {artifact_dir}")


def evidence_items_with_qid(result: AnswerResult) -> list[dict]:
    """生成审计证据条目；每条证据显式带 qid，便于外部校验和人工追踪。"""
    rows = []
    for item in result.evidence:
        row = item.to_dict()
        row["qid"] = result.qid
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
