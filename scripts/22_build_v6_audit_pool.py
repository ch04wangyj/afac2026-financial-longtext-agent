"""脚本 22：用证据契约构建 V6 高风险复核池。

该脚本只做离线分析，不修改任何答案。它比较官方 V5 基线与一个或多个
候选结果，并按证据缺失、冲突、否定/全称断言和模型自相矛盾排序。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.io.jsonl import read_jsonl, write_json
from agent.reasoning.evidence_contract import build_evidence_contracts
from agent.reasoning.precise_verifier import _answer_from_checks
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 V6 证据完备性复核池。")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--include-unchanged",
        action="store_true",
        help="同时输出候选答案未变化但证据风险较高的题目。",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    questions = {
        question.qid: question for question in load_questions(settings.questions_root)
    }
    baseline = {
        row.qid: row
        for row in (
            AnswerResult.from_dict(data) for data in read_jsonl(args.baseline)
        )
    }
    candidate_rows = [
        AnswerResult.from_dict(data)
        for path in args.candidate
        for data in read_jsonl(path)
    ]
    duplicate_qids = sorted(
        qid for qid, count in Counter(row.qid for row in candidate_rows).items() if count > 1
    )
    if duplicate_qids:
        raise ValueError(f"候选文件包含重复 qid: {duplicate_qids}")

    rows: list[dict] = []
    for candidate in candidate_rows:
        question = questions.get(candidate.qid)
        base = baseline.get(candidate.qid)
        if question is None or base is None:
            continue
        changed = candidate.answer != base.answer
        contracts = build_evidence_contracts(question, candidate.evidence)
        contract_rows = {
            key: contract.to_dict() for key, contract in contracts.items()
        }
        needs_review = [
            key for key, contract in contracts.items() if contract.needs_review
        ]
        risk_tags = sorted(
            {
                tag
                for contract in contracts.values()
                for tag in contract.risk_tags
            }
        )
        answer_from_checks = _answer_from_checks(candidate.raw_response, question)
        checks_disagree = bool(
            answer_from_checks and answer_from_checks != candidate.answer
        )
        risk_score = _risk_score(
            changed=changed,
            needs_review_count=len(needs_review),
            risk_tags=risk_tags,
            checks_disagree=checks_disagree,
            confidence=candidate.confidence,
        )
        if not args.include_unchanged and not changed:
            continue
        rows.append(
            {
                "qid": candidate.qid,
                "domain": question.domain,
                "baseline_answer": base.answer,
                "candidate_answer": candidate.answer,
                "changed": changed,
                "confidence": candidate.confidence,
                "answer_from_checks": answer_from_checks,
                "checks_disagree": checks_disagree,
                "needs_review_options": needs_review,
                "risk_tags": risk_tags,
                "risk_score": risk_score,
                "contracts": contract_rows,
            }
        )

    rows.sort(key=lambda row: (-row["risk_score"], row["qid"]))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit_pool.json", rows)
    write_json(
        output_dir / "audit_summary.json",
        {
            "baseline": str(args.baseline.resolve()),
            "candidates": [str(path.resolve()) for path in args.candidate],
            "question_count": len(rows),
            "changed_count": sum(1 for row in rows if row["changed"]),
            "checks_disagree_count": sum(1 for row in rows if row["checks_disagree"]),
            "domain_counts": dict(Counter(row["domain"] for row in rows)),
            "risk_tag_counts": dict(
                Counter(tag for row in rows for tag in row["risk_tags"])
            ),
        },
    )
    (output_dir / "audit_pool.md").write_text(
        _render_markdown(rows),
        encoding="utf-8",
    )
    print(f"V6 审计池题目数: {len(rows)}")
    print(f"输出目录: {output_dir}")


def _risk_score(
    *,
    changed: bool,
    needs_review_count: int,
    risk_tags: list[str],
    checks_disagree: bool,
    confidence: float,
) -> float:
    """风险分只用于排序，不作为自动改答案依据。"""
    tag_weights = {
        "absence_claim": 2.0,
        "evidence_conflict": 1.8,
        "financial_scope_ambiguity": 2.5,
        "universal_scope": 1.2,
        "comparison": 1.0,
        "negative_claim": 1.0,
        "compound_claim": 0.8,
    }
    score = 3.0 if changed else 0.0
    score += min(6.0, needs_review_count * 1.5)
    score += sum(tag_weights.get(tag, 0.5) for tag in risk_tags)
    score += 2.0 if checks_disagree else 0.0
    if confidence < 0.6:
        score += 1.0
    return round(score, 3)


def _render_markdown(rows: list[dict]) -> str:
    lines = [
        "# V6 证据契约审计池",
        "",
        "该列表只用于排序复核，不能直接视为答案变更清单。",
        "",
        "| QID | 领域 | V5 | 候选 | 风险分 | 待复核选项 | 风险标签 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {qid} | {domain} | {baseline_answer} | {candidate_answer} | "
            "{risk_score:.1f} | {options} | {tags} |".format(
                **row,
                options=",".join(row["needs_review_options"]) or "-",
                tags=",".join(row["risk_tags"]) or "-",
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
