"""脚本 20：用 V14 局部运行替换 Token，并保守融合答案。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evaluation.selective_merge import merge_candidate_with_baseline
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge V14 layout candidate with official V13 baseline.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, default=ROOT / "configs" / "v14_layout_reviews.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)]
    candidate = [AnswerResult.from_dict(row) for row in read_jsonl(args.candidate)]
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    merged = merge_candidate_with_baseline(baseline, candidate, reviews)
    write_jsonl(args.output, (row.to_dict() for row in merged))

    changed = [row.qid for row in merged if row.answer != next(item.answer for item in baseline if item.qid == row.qid)]
    total_tokens = sum(row.token_usage.total_tokens for row in merged)
    print(f"wrote {len(merged)} merged results, changes={changed}, total_tokens={total_tokens} -> {args.output}")


if __name__ == "__main__":
    main()
