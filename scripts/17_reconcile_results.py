"""脚本 17：保守融合 V3 与 V2，并只审计强证据变化。"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.llm.qwen_client import QwenClient
from agent.reasoning.result_reconciler import ReconcileConfig, ResultReconciler
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="保守融合 V3 结果与 V2 基线。")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["AFAC_QWEN_MODEL"] = args.model
    settings = Settings.from_env()
    questions = {question.qid: question for question in load_questions(settings.questions_root)}
    current_rows = {row["qid"]: AnswerResult.from_dict(row) for row in read_jsonl(args.results)}
    baseline_rows = {row["qid"]: AnswerResult.from_dict(row) for row in read_jsonl(args.baseline)}
    missing = sorted(set(questions) - set(current_rows) | (set(questions) - set(baseline_rows)))
    if missing:
        raise ValueError(f"Missing qids in input results: {missing}")

    reconciler = ResultReconciler(
        QwenClient(settings, dry_run=args.dry_run),
        ReconcileConfig(enable_thinking=args.thinking),
    )

    def reconcile(qid: str):
        result = reconciler.reconcile(questions[qid], current_rows[qid], baseline_rows[qid])
        print(
            f"[{qid}] {current_rows[qid].answer}->{result.answer} "
            f"decision={result.metadata.get('reconcile_decision')}",
            flush=True,
        )
        return qid, result

    output: dict[str, AnswerResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(reconcile, qid): qid for qid in questions}
        for future in as_completed(futures):
            qid, result = future.result()
            output[qid] = result

    ordered = [output[qid].to_dict() for qid in questions]
    write_jsonl(args.output, ordered)
    print(f"wrote {len(ordered)} reconciled results -> {args.output}")


if __name__ == "__main__":
    main()
