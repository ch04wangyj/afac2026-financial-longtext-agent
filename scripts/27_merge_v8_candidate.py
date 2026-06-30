"""脚本 27：在 V7 官网基线上合并指定的 V8 针对性复核组。"""

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
    parser = argparse.ArgumentParser(description="生成 V8 针对性复核候选。")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs" / "releases" / "v7" / "answer_results.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "v8_targeted_reviews.json",
    )
    parser.add_argument(
        "--include-group",
        action="append",
        required=True,
        help="纳入一个复核组；可重复传入。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_results(args.baseline)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    groups = dict(payload.get("groups", {}))
    unknown = sorted(set(args.include_group) - set(groups))
    if unknown:
        raise KeyError(f"未知复核组: {unknown}; 可用值: {sorted(groups)}")

    reviews: dict[str, dict] = {}
    candidates_by_qid: dict[str, AnswerResult] = {}
    for group_name in args.include_group:
        group = groups[group_name]
        candidate_path = ROOT / str(group["candidate_result"])
        source_candidates = {row.qid: row for row in _load_results(candidate_path)}
        for qid, review in dict(group.get("reviews", {})).items():
            if qid in reviews:
                raise ValueError(f"复核组之间包含重复题号: {qid}")
            if qid not in source_candidates:
                raise KeyError(f"{candidate_path} 缺少复核题 {qid}")
            reviews[qid] = review
            candidates_by_qid[qid] = source_candidates[qid]

    merged = merge_candidate_with_baseline(
        baseline,
        list(candidates_by_qid.values()),
        reviews,
        baseline_label="v7_official_84.3124",
        candidate_label="v8_targeted_review",
    )
    write_jsonl(args.output, (row.to_dict() for row in merged))

    baseline_answers = {row.qid: row.answer for row in baseline}
    changed = [row.qid for row in merged if row.answer != baseline_answers[row.qid]]
    baseline_tokens = sum(row.token_usage.total_tokens for row in baseline)
    merged_tokens = sum(row.token_usage.total_tokens for row in merged)
    print(f"复核组: {args.include_group}")
    print(f"变化题号: {changed}")
    print(f"V7 Token: {baseline_tokens:,}")
    print(f"候选 Token: {merged_tokens:,} ({merged_tokens - baseline_tokens:+,})")
    print(f"输出: {args.output.resolve()}")


def _load_results(path: Path) -> list[AnswerResult]:
    rows = [AnswerResult.from_dict(row) for row in read_jsonl(path)]
    if not rows:
        raise RuntimeError(f"结果文件为空: {path}")
    return rows


if __name__ == "__main__":
    main()
