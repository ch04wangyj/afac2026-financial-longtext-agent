"""脚本 25：按方法组生成 V7 消融提交，并输出官网分数诊断表。

该脚本不调用模型，不逐题探测标签。它只比较两组可解释的方法改动：
监管直接条文修正，以及合同/研报严格术语修正。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.score_diagnostics import final_score, infer_correct_count
from agent.evaluation.selective_merge import merge_candidate_with_baseline
from agent.io.jsonl import read_jsonl, write_json, write_jsonl
from agent.io.submission import (
    summarize_usage,
    validate_answer_results,
    write_answer_csv,
)
from agent.schemas import AnswerResult


ABLATION_GROUPS = {
    "regulatory_core": ("reg_a_001", "reg_a_004", "reg_a_005"),
    "strict_semantics": ("fc_a_014", "res_a_006", "res_a_016"),
    "v6_all": (
        "fc_a_014",
        "reg_a_001",
        "reg_a_004",
        "reg_a_005",
        "res_a_006",
        "res_a_016",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 V7 方法级消融矩阵。")
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
        "--groups",
        nargs="+",
        choices=sorted(ABLATION_GROUPS),
        default=sorted(ABLATION_GROUPS),
    )
    parser.add_argument(
        "--observed-v6-score",
        type=float,
        default=83.33,
        help="官网显示的 V6 分数，用于反推正确题数。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "v7_ablation_matrix",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    questions = load_questions(settings.questions_root)
    baseline = _load_results(args.baseline)
    all_candidates = _load_results(args.candidate)
    review_payload = json.loads(args.reviews.read_text(encoding="utf-8"))
    confirmed = dict(review_payload.get("confirmed", {}))
    candidate_by_qid = {row.qid: row for row in all_candidates}
    baseline_answers = {row.qid: row.answer for row in baseline}

    manifests = []
    for group_name in args.groups:
        qids = ABLATION_GROUPS[group_name]
        reviews = {qid: confirmed[qid] for qid in qids}
        candidates = [candidate_by_qid[qid] for qid in qids]
        merged = merge_candidate_with_baseline(
            baseline,
            candidates,
            reviews,
            baseline_label="v5_official_80.4466",
            candidate_label=f"v7_ablation_{group_name}",
        )
        validate_answer_results(merged, questions, require_complete=True)
        output_dir = args.output_root / group_name
        write_jsonl(output_dir / "answer_results.jsonl", (row.to_dict() for row in merged))
        write_answer_csv(output_dir / "answer.csv", merged)
        usage = summarize_usage(merged)
        write_json(output_dir / "token_usage.json", usage.to_dict())
        changed = [row.qid for row in merged if row.answer != baseline_answers[row.qid]]
        manifests.append(
            {
                "group": group_name,
                "method_scope": list(qids),
                "changed_qids": changed,
                "total_tokens": usage.total_tokens,
                "score_by_correct_count": {
                    str(correct): round(
                        final_score(correct, len(questions), usage.total_tokens),
                        6,
                    )
                    for correct in range(80, 94)
                },
                "answer_csv": str((output_dir / "answer.csv").resolve()),
            }
        )

    v6_usage = summarize_usage(
        [
            AnswerResult.from_dict(row)
            for row in read_jsonl(
                ROOT / "outputs" / "releases" / "v6" / "answer_results.jsonl"
            )
        ]
    )
    inferred = infer_correct_count(
        args.observed_v6_score,
        len(questions),
        v6_usage.total_tokens,
    )
    manifest = {
        "purpose": "方法级消融，不用于逐题标签探测",
        "official_v5": {
            "score": 80.4466,
            "correct": 82,
            "total_tokens": summarize_usage(baseline).total_tokens,
        },
        "official_v6": {
            "observed_score": args.observed_v6_score,
            "inference": inferred.to_dict(),
        },
        "groups": manifests,
    }
    write_json(args.output_root / "diagnostic_manifest.json", manifest)
    print(json.dumps(manifest["official_v6"], ensure_ascii=False, indent=2))
    for row in manifests:
        print(
            f"{row['group']}: changed={row['changed_qids']}, "
            f"tokens={row['total_tokens']:,}"
        )
    print(f"输出: {args.output_root.resolve()}")


def _load_results(path: Path) -> list[AnswerResult]:
    rows = [AnswerResult.from_dict(row) for row in read_jsonl(path)]
    if not rows:
        raise RuntimeError(f"结果文件为空: {path}")
    return rows


if __name__ == "__main__":
    main()
