"""脚本 04：根据 answer_results.jsonl 生成提交和审计产物。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.io.jsonl import read_jsonl, write_json
from agent.io.submission import summarize_usage, write_answer_csv
from agent.schemas import AnswerResult


def main() -> None:
    """写出 answer.csv、evidence.json 和 token_usage.json。"""
    parser = argparse.ArgumentParser(description="Create answer.csv, evidence.json, and token_usage.json.")
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    results_path = args.results or settings.outputs_dir / "answer_results.jsonl"
    results = [AnswerResult.from_dict(row) for row in read_jsonl(results_path)]
    if not results:
        raise RuntimeError(f"No results found: {results_path}")

    write_answer_csv(settings.outputs_dir / "answer.csv", results)
    write_json(
        settings.outputs_dir / "evidence.json",
        {result.qid: evidence_items_with_qid(result) for result in results},
    )
    usage = summarize_usage(results)
    write_json(settings.outputs_dir / "token_usage.json", usage.to_dict())
    print(f"wrote submission artifacts to {settings.outputs_dir}")


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
