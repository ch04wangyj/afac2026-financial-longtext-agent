"""脚本 23：用局部候选运行替换 Token，并保守融合答案。

保守融合策略：
- 候选结果的 Token usage 全量继承（V15 的在线 Token）
- 答案变化默认回退 V14 官网已验证答案
- 只有 reviews 配置中显式复核的题才改答案
- 保持 100 题完整性
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evaluation.selective_merge import merge_candidate_with_baseline
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local candidates with official V14 baseline.")
    parser.add_argument("--baseline", type=Path, required=True, help="V14 official answer_results.jsonl")
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个互不重复的局部候选 answer_results.jsonl",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "configs" / "v16_structure_reviews.json",
        help="逐题直接原文复核答案",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)]
    candidate = [
        AnswerResult.from_dict(row)
        for candidate_path in args.candidate
        for row in read_jsonl(candidate_path)
    ]
    candidate_qids = [row.qid for row in candidate]
    duplicate_qids = sorted(qid for qid, count in Counter(candidate_qids).items() if count > 1)
    if duplicate_qids:
        raise ValueError(f"Duplicate qids across candidate files: {duplicate_qids}")

    reviews = {}
    if args.reviews.exists():
        reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    else:
        print(f"  No reviews file at {args.reviews}, all differences will fallback to V14.")

    merged = merge_candidate_with_baseline(baseline, candidate, reviews)
    # 统一写入候选版本名，避免局部运行的旧策略名污染审计结果。
    for row in merged:
        row.metadata["strategy"] = "v16_structure_selected_truth_merge"

    write_jsonl(args.output, (row.to_dict() for row in merged))

    baseline_map = {row.qid: row for row in baseline}
    changed = []
    for row in merged:
        if row.answer != baseline_map[row.qid].answer:
            changed.append(row.qid)
    total_tokens = sum(row.token_usage.total_tokens for row in merged)
    baseline_tokens = sum(row.token_usage.total_tokens for row in baseline)

    print(f"{'=' * 60}")
    print("V16 Structure-Selected-Truth Merge Result")
    print(f"{'=' * 60}")
    print(f"  Total questions: {len(merged)}")
    print(f"  Answer changes vs V14: {len(changed)}")
    if changed:
        print(f"    Changed qids: {changed}")
    print(f"  V14 baseline tokens: {baseline_tokens:,}")
    print(f"  V16 merged tokens: {total_tokens:,}")
    delta = total_tokens - baseline_tokens
    print(f"  Token delta: {delta:+,} ({delta / baseline_tokens * 100:+.1f}%)")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
