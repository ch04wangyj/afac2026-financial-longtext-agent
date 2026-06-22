"""脚本 14：运行 V12 文档级高召回验证器。"""

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
from agent.index.bm25 import BM25SearchIndex
from agent.io.jsonl import append_jsonl, read_jsonl, write_json, write_jsonl
from agent.llm.qwen_client import QwenClient
from agent.reasoning.exhaustive_verifier import ExhaustiveVerifier, ExhaustiveVerifierConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V12 exhaustive document verifier.")
    parser.add_argument("--qids", nargs="*", default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=52_000)
    args = parser.parse_args()

    if args.model:
        os.environ["AFAC_QWEN_MODEL"] = args.model
    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = load_questions(settings.questions_root, domains=args.domains)
    questions = _select_questions(questions, args.qids, args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.index or settings.index_dir / "bm25_index.pkl"
    index = BM25SearchIndex.load(index_path)
    verifier = ExhaustiveVerifier(
        index,
        QwenClient(settings, dry_run=args.dry_run),
        ExhaustiveVerifierConfig(
            audit_enabled=args.audit,
            max_context_chars=args.max_context_chars,
        ),
    )
    manifest = [
        {
            "qid": question.qid,
            "domain": question.domain,
            "answer_format": question.answer_format,
            "doc_ids": question.doc_ids,
        }
        for question in questions
    ]
    write_json(output_dir / "sample_manifest.json", manifest)

    out_path = output_dir / "answer_results.jsonl"
    existing = _load_existing(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    todo = [question for question in questions if question.qid not in existing]

    def solve(question):
        print(f"[{question.qid}] exhaustive solving", flush=True)
        return question.qid, verifier.solve(question).to_dict()

    solved = dict(existing)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(solve, question): question.qid for question in todo}
        for future in as_completed(futures):
            qid, row = future.result()
            solved[qid] = row
            append_jsonl(out_path, row)
            print(f"[{qid}] checkpointed answer={row['answer']}", flush=True)

    ordered = [solved[question.qid] for question in questions if question.qid in solved]
    write_jsonl(out_path, ordered)
    print(f"wrote {len(ordered)} exhaustive results -> {out_path}", flush=True)


def _select_questions(questions, qids: list[str] | None, limit: int):
    if qids:
        by_qid = {question.qid: question for question in questions}
        missing = [qid for qid in qids if qid not in by_qid]
        if missing:
            raise KeyError(f"Unknown qids: {missing}")
        selected = [by_qid[qid] for qid in qids]
    else:
        selected = list(questions)
    return selected[:limit] if limit else selected


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["qid"]: row for row in read_jsonl(path) if row.get("qid")}


if __name__ == "__main__":
    main()
