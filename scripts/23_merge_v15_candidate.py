"""脚本 23：用 V15 局部运行替换 Token，并保守融合答案。

保守融合策略：
- 候选结果的 Token usage 全量继承（V15 的在线 Token）
- 答案变化默认回退 V14 官网已验证答案
- 只有 configs/v15_layout_reviews.json 中显式复核的题才改答案
- 保持 100 题完整性
"""

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
    parser = argparse.ArgumentParser(description="Merge V15 candidate with official V14 baseline.")
    parser.add_argument("--baseline", type=Path, required=True, help="V14 official answer_results.jsonl")
    parser.add_argument("--candidate", type=Path, required=True, help="V15 candidate answer_results.jsonl")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "configs" / "v15_layout_reviews.json",
        help="V15 reviewed answer overrides",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)]
    candidate = [AnswerResult.from_dict(row) for row in read_jsonl(args.candidate)]

    reviews = {}
    if args.reviews.exists():
        reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    else:
        print(f"  No reviews file at {args.reviews}, all differences will fallback to V14.")

    merged = merge_candidate_with_baseline(baseline, candidate, reviews)
    # 更新 strategy 名称为 V15
    for row in merged:
        row.metadata["strategy"] = "v15_layout_selective_merge"

    write_jsonl(args.output, (row.to_dict() for row in merged))

    baseline_map = {row.qid: row for row in baseline}
    changed = []
    for row in merged:
        if row.answer != baseline_map[row.qid].answer:
            changed.append(row.qid)
    total_tokens = sum(row.token_usage.total_tokens for row in merged)
    baseline_tokens = sum(row.token_usage.total_tokens for row in baseline)

    print(f"{'=' * 60}")
    print(f"V15 Conservative Merge Result")
    print(f"{'=' * 60}")
    print(f"  Total questions: {len(merged)}")
    print(f"  Answer changes vs V14: {len(changed)}")
    if changed:
        print(f"    Changed qids: {changed}")
    print(f"  V14 baseline tokens: {baseline_tokens:,}")
    print(f"  V15 merged tokens: {total_tokens:,}")
    delta = total_tokens - baseline_tokens
    print(f"  Token delta: {delta:+,} ({delta / baseline_tokens * 100:+.1f}%)")
    print(f"  Output: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
