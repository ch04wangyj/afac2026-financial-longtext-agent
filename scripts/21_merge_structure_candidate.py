"""脚本 21：将 V5 局部候选与 V4 官方基线做保守、可审计融合。

候选结果提供真实在线 Token 和证据；未进入人工复核配置的答案变化一律回退 V4。
该策略把“召回改善”和“答案改动”解耦，防止全量重判造成不可控漂移。
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
    parser = argparse.ArgumentParser(description="将 V5 局部候选与 V4 官方基线保守融合。")
    parser.add_argument("--baseline", type=Path, required=True, help="V4 官方 answer_results.jsonl")
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个互不重复的 V5 局部候选 answer_results.jsonl",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "configs" / "v5_structure_reviews.json",
        help="逐题直接原文复核答案",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)]
    candidates = [
        AnswerResult.from_dict(row)
        for candidate_path in args.candidate
        for row in read_jsonl(candidate_path)
    ]
    counts = Counter(row.qid for row in candidates)
    duplicate_qids = sorted(qid for qid, count in counts.items() if count > 1)
    if duplicate_qids:
        raise ValueError(f"候选文件包含重复 qid: {duplicate_qids}")

    if not args.reviews.exists():
        raise FileNotFoundError(f"复核配置不存在: {args.reviews}")
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    merged = merge_candidate_with_baseline(
        baseline,
        candidates,
        reviews,
        baseline_label="v4_layout",
        candidate_label="v5_structure",
    )
    write_jsonl(args.output, (row.to_dict() for row in merged))

    baseline_answers = {row.qid: row.answer for row in baseline}
    changed = [row.qid for row in merged if row.answer != baseline_answers[row.qid]]
    total_tokens = sum(row.token_usage.total_tokens for row in merged)
    baseline_tokens = sum(row.token_usage.total_tokens for row in baseline)

    print("=" * 60)
    print("V5 结构导航保守融合结果")
    print("=" * 60)
    print(f"题目数: {len(merged)}")
    print(f"相对 V4 答案变化: {len(changed)}")
    print(f"变化题号: {changed}")
    print(f"V4 Token: {baseline_tokens:,}")
    print(f"V5 Token: {total_tokens:,}")
    print(f"Token 变化: {total_tokens - baseline_tokens:+,}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
