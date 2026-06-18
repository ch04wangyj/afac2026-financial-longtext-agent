"""脚本 06：按领域抽样运行 smoke 测试。

用于真实调用前后快速覆盖五个领域，输出结构与 03_run_questions.py 保持一致。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.data.questions import load_questions
from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.io.output_layout import choose_output_dir
from agent.llm.qwen_client import QwenClient
from agent.reasoning.solver import Solver
from agent.retrieve.retriever import Retriever


DEFAULT_DOMAINS = ["financial_contracts", "financial_reports", "insurance", "regulatory", "research"]


def main() -> None:
    """每个领域选前 N 道题，运行完整求解链路。"""
    parser = argparse.ArgumentParser(description="Run a per-domain smoke test.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Qwen.")
    parser.add_argument("--per-domain", type=int, default=1, help="Number of questions per domain.")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS, help="Domains to include.")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing answer_results.jsonl.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = _select_questions(load_questions(settings.questions_root, domains=args.domains), args.per_domain)
    run_dir = choose_output_dir(
        settings,
        run_scope="test",
        run_name="smoke",
        strategy=settings.retrieval_strategy,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    index_path = args.index or settings.index_dir / "bm25_index.pkl"
    index = BM25SearchIndex.load(index_path)
    doc_index_path = settings.index_dir / "document_bm25_index.pkl"
    doc_index = DocumentSearchIndex.load(doc_index_path) if doc_index_path.exists() else None
    retriever = Retriever(
        index,
        doc_index=doc_index,
        top_k_per_query=settings.top_k_retrieval,
        fused_top_k=settings.top_k_retrieval,
        strategy=settings.retrieval_strategy,
        blind_top_docs=settings.blind_top_docs,
    )
    compressor = RuleEvidenceCompressor(max_chars=settings.max_evidence_chars, top_k=settings.top_k_evidence)
    solver = Solver(retriever, compressor, QwenClient(settings, dry_run=args.dry_run))

    out_path = run_dir / "answer_results.jsonl"
    existing_by_qid = _load_existing_results(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    results = []
    todo_questions = [question for question in questions if question.qid not in existing_by_qid]
    if existing_by_qid:
        for idx, question in enumerate(questions, start=1):
            if question.qid in existing_by_qid:
                print(f"[{idx}/{len(questions)}] skip {question.domain}/{question.qid} from checkpoint", flush=True)
                results.append(existing_by_qid[question.qid])

    def solve_question(question):
        print(f"solving {question.domain}/{question.qid}", flush=True)
        return question.qid, solver.solve(question).to_dict()

    with ThreadPoolExecutor(max_workers=settings.question_workers) as executor:
        futures = {executor.submit(solve_question, question): question.qid for question in todo_questions}
        pending = []
        for future in as_completed(futures):
            pending.append(future.result())

    solved_by_qid = {qid: row for qid, row in pending}
    results.extend(solved_by_qid[question.qid] for question in todo_questions)
    write_jsonl(out_path, results)
    print(f"wrote {len(results)} smoke results -> {out_path}", flush=True)


def _select_questions(questions, per_domain: int):
    """按领域保留前 per_domain 道题。"""
    grouped = defaultdict(list)
    for question in questions:
        if len(grouped[question.domain]) < per_domain:
            grouped[question.domain].append(question)
    selected = []
    for domain in DEFAULT_DOMAINS:
        selected.extend(grouped.get(domain, []))
    return selected


def _load_existing_results(path: Path) -> dict[str, dict]:
    """读取已完成结果，续跑时用 qid 去重。"""
    if not path.exists():
        return {}
    return {row["qid"]: row for row in read_jsonl(path) if row.get("qid")}


if __name__ == "__main__":
    main()
