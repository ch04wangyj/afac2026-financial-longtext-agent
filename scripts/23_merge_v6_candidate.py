"""脚本 23：将 V6 证据契约候选与 V5 官网基线做分层、可审计融合。

默认只应用 confirmed 复核。可选 probe 必须显式传入，防止尚未得到原文闭环的
模型差异混入正式提交。仅纳入复核题对应的候选行，未变化题继续使用 V5 证据
与 Token，避免一次探索性全量运行抬高正式提交的 Token 统计。
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
    parser = argparse.ArgumentParser(description="将 V6 局部候选与 V5 官网基线保守融合。")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs" / "releases" / "v5" / "answer_results.jsonl",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "outputs" / "v6_full_candidate" / "answer_results.jsonl",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "configs" / "v6_evidence_reviews.json",
    )
    parser.add_argument(
        "--include-probe",
        action="append",
        default=[],
        help="显式纳入指定探测组；可重复传入。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)]
    all_candidates = [AnswerResult.from_dict(row) for row in read_jsonl(args.candidate)]
    review_payload = json.loads(args.reviews.read_text(encoding="utf-8"))
    reviews = _select_reviews(review_payload, args.include_probe)

    candidate_by_qid = {row.qid: row for row in all_candidates}
    missing_candidates = sorted(set(reviews) - set(candidate_by_qid))
    if missing_candidates:
        raise KeyError(f"V6 候选缺少复核题: {missing_candidates}")
    selected_candidates = [candidate_by_qid[qid] for qid in reviews]

    merged = merge_candidate_with_baseline(
        baseline,
        selected_candidates,
        reviews,
        baseline_label="v5_official_80.4466",
        candidate_label="v6_evidence_contract",
    )
    write_jsonl(args.output, (row.to_dict() for row in merged))

    baseline_answers = {row.qid: row.answer for row in baseline}
    changed = [row.qid for row in merged if row.answer != baseline_answers[row.qid]]
    baseline_tokens = sum(row.token_usage.total_tokens for row in baseline)
    merged_tokens = sum(row.token_usage.total_tokens for row in merged)
    print(f"题目数: {len(merged)}")
    print(f"复核项: {len(reviews)}")
    print(f"相对 V5 答案变化: {len(changed)}")
    print(f"变化题号: {changed}")
    print(f"V5 Token: {baseline_tokens:,}")
    print(f"V6 Token: {merged_tokens:,} ({merged_tokens - baseline_tokens:+,})")
    print(f"输出: {args.output}")


def _select_reviews(payload: dict, probe_names: list[str]) -> dict[str, dict]:
    """合并 confirmed 与显式指定的 probe，并拒绝重复题号。"""
    reviews = dict(payload.get("confirmed", {}))
    available_probes = dict(payload.get("probes", {}))
    unknown = sorted(set(probe_names) - set(available_probes))
    if unknown:
        raise KeyError(f"未知 probe: {unknown}; 可用值: {sorted(available_probes)}")

    for name in probe_names:
        for qid, review in available_probes[name].items():
            if qid in reviews:
                raise ValueError(f"probe {name} 与已有复核重复: {qid}")
            reviews[qid] = review
    return reviews


if __name__ == "__main__":
    main()
