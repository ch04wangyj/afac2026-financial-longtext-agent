"""脚本 16：运行 V3 原子证据或 V5 结构导航精确验证器。"""

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
from agent.reasoning.precise_verifier import PreciseVerifier, PreciseVerifierConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 V3/V5 精确证据验证器。")
    parser.add_argument("--qids", nargs="*", default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-max")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    parser.add_argument("--strategy-name", default="v3_atomic_precise")
    parser.add_argument(
        "--structure-navigation",
        action="store_true",
        help="启用无 embedding 的章节/页面导航增补召回。",
    )
    parser.add_argument(
        "--assemble-from-checks",
        action="store_true",
        help="按逐项 truth 程序化组装答案，要求 truth 表示是否应被题干选中。",
    )
    parser.add_argument(
        "--evidence-contract",
        action="store_true",
        help="启用 V6 选项级文档、谓词和数值端点完备性约束。",
    )
    parser.add_argument(
        "--numeric-verifier",
        action="store_true",
        help="启用 V6 财报白名单数值账本与确定性比较/增长复算。",
    )
    args = parser.parse_args()

    os.environ["AFAC_QWEN_MODEL"] = args.model
    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = _select_questions(load_questions(settings.questions_root, domains=args.domains), args.qids, args.limit)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.index or settings.processed_dir / "v3_atomic" / "bm25_index.pkl"
    verifier = PreciseVerifier(
        BM25SearchIndex.load(index_path),
        QwenClient(settings, dry_run=args.dry_run),
        PreciseVerifierConfig(
            audit_enabled=args.audit,
            enable_thinking=not args.no_thinking,
            max_context_chars=args.max_context_chars,
            strategy_name=args.strategy_name,
            enable_structure_navigation=args.structure_navigation,
            enable_evidence_contract=args.evidence_contract,
            enable_numeric_verifier=args.numeric_verifier,
            assemble_answer_from_checks=args.assemble_from_checks,
        ),
    )
    write_json(
        output_dir / "sample_manifest.json",
        [
            {
                "qid": question.qid,
                "domain": question.domain,
                "answer_format": question.answer_format,
                "doc_ids": question.doc_ids,
            }
            for question in questions
        ],
    )

    out_path = output_dir / "answer_results.jsonl"
    existing = _load_existing(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()
    todo = [question for question in questions if question.qid not in existing]

    def solve(question):
        print(f"[{question.qid}] {args.strategy_name} solving", flush=True)
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
    print(f"wrote {len(ordered)} precise results -> {out_path}", flush=True)


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
