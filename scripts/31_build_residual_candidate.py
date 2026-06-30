"""脚本 31：从完整官网基线构建分层残差修正候选。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evaluation.review_overrides import apply_review_overrides
from agent.evaluation.residual_profiles import resolve_profile_reviews
from agent.evaluation.score_diagnostics import final_score
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="生成分层残差修正结果。")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs" / "releases" / "v7" / "answer_results.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "v9_residual_reviews.json",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--baseline-correct",
        type=int,
        default=86,
        help="基线的官网正确题数，仅用于打印全部变更命中时的理论上界。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = [
        AnswerResult.from_dict(row)
        for row in read_jsonl(args.baseline)
        if row.get("qid") != "summary"
    ]
    if len(baseline) != 100:
        raise RuntimeError(f"候选基线必须完整包含 100 题: {args.baseline}")

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    reviews = resolve_profile_reviews(payload.get("profiles", {}), args.profile)
    candidate = apply_review_overrides(baseline, reviews)
    write_jsonl(args.output, (row.to_dict() for row in candidate))

    baseline_answers = {row.qid: row.answer for row in baseline}
    changed = [
        f"{row.qid}: {baseline_answers[row.qid]} -> {row.answer}"
        for row in candidate
        if row.answer != baseline_answers[row.qid]
    ]
    total_tokens = sum(row.token_usage.total_tokens for row in candidate)
    # 这里只报告全部变化都由错改对时的上界，不把理论值写成实测成绩。
    assumed_correct = args.baseline_correct + len(changed)
    print(f"配置: {args.profile}")
    print("变化:")
    for item in changed:
        print(f"  {item}")
    print(f"Token: {total_tokens:,}")
    print(
        f"全部 {len(changed)} 处均修正时: "
        f"{assumed_correct}/100, 理论综合分 {final_score(assumed_correct, 100, total_tokens):.6f}"
    )
    print(f"输出: {args.output.resolve()}")
if __name__ == "__main__":
    main()
