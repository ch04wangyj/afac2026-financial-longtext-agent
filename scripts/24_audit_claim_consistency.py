"""脚本 24：输出跨题重复事实的答案冲突清单，供逐条原文复核。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.evaluation.claim_consistency import build_claim_records, find_claim_conflicts
from agent.io.jsonl import read_jsonl, write_json
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="审计同文档近重复断言的答案一致性。")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--support-results",
        type=Path,
        default=None,
        help="可选的 V6 结果，用其证据契约限定每个选项实际对应的源文档。",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-similarity", type=float, default=0.78)
    parser.add_argument(
        "--allow-partial-doc-overlap",
        action="store_true",
        help="探索模式：允许题目仅共享部分文档；输出误报会明显增加。",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    questions = load_questions(settings.questions_root)
    answers = [AnswerResult.from_dict(row) for row in read_jsonl(args.results)]
    support_results = (
        [AnswerResult.from_dict(row) for row in read_jsonl(args.support_results)]
        if args.support_results
        else None
    )
    records = build_claim_records(questions, answers, support_results=support_results)
    conflicts = find_claim_conflicts(
        records,
        min_similarity=args.min_similarity,
        require_same_doc_set=not args.allow_partial_doc_overlap,
    )
    write_json(args.output, [item.to_dict() for item in conflicts])

    print(f"事实节点: {len(records)}")
    print(f"冲突候选: {len(conflicts)}")
    for conflict in conflicts:
        print(
            f"{conflict.left.qid}/{conflict.left.option}={conflict.left.selected} <> "
            f"{conflict.right.qid}/{conflict.right.option}={conflict.right.selected} "
            f"similarity={conflict.similarity:.3f}"
        )
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
