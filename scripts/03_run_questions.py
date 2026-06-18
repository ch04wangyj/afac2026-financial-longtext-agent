"""脚本 03：运行检索 + 压缩 + Qwen 作答闭环。"""

from __future__ import annotations

import argparse
import os
import sys
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
from agent.runtime.parallel import parallel_map_ordered


def main() -> None:
    """执行题目求解；真实调用前建议用 --limit 控制规模。"""
    parser = argparse.ArgumentParser(description="Run retrieval + Qwen answering for A group questions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Qwen; emit deterministic dummy answers.")
    parser.add_argument("--a-board-quality", action="store_true", help="Enable A-board coverage gate, option matrix, and financial calculator.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions for smoke tests.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domains to include.")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing answer_results.jsonl.")
    args = parser.parse_args()

    if args.a_board_quality:
        _enable_a_board_quality_mode()

    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = load_questions(settings.questions_root, domains=args.domains)
    if args.limit:
        questions = questions[: args.limit]

    is_full_a100 = not args.dry_run and not args.limit and args.domains is None and len(questions) == 100
    run_scope = "a100" if is_full_a100 else "test"
    run_name = "full100" if is_full_a100 else _test_run_name(args.limit, args.domains)
    run_dir = choose_output_dir(
        settings,
        run_scope=run_scope,
        run_name=run_name,
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
    llm = QwenClient(settings, dry_run=args.dry_run)
    solver = Solver(retriever, compressor, llm)

    out_path = run_dir / "answer_results.jsonl"
    existing_by_qid = _load_existing_results(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    rows = []
    todo_questions = [question for question in questions if question.qid not in existing_by_qid]
    if existing_by_qid:
        for idx, question in enumerate(questions, start=1):
            if question.qid in existing_by_qid:
                print(f"[{idx}/{len(questions)}] skip {question.qid} from checkpoint", flush=True)
                rows.append(existing_by_qid[question.qid])

    def solve_question(question):
        print(f"solving {question.qid}", flush=True)
        return question.qid, solver.solve(question).to_dict()

    with ThreadPoolExecutor(max_workers=settings.question_workers) as executor:
        futures = {executor.submit(solve_question, question): question.qid for question in todo_questions}
        pending = []
        for future in as_completed(futures):
            pending.append(future.result())

    solved_by_qid = {qid: row for qid, row in pending}
    rows.extend(solved_by_qid[question.qid] for question in todo_questions)
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} answer results -> {out_path}", flush=True)


def _enable_a_board_quality_mode() -> None:
    """通过环境变量打开 A 榜质量模式，避免手改 YAML。"""
    os.environ["AFAC_A_BOARD_OPTION_MATRIX_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_COVERAGE_GATE_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY"] = "false"



def _test_run_name(limit: int, domains: list[str] | None) -> str:
    if limit:
        return f"run_questions_limit{limit}"
    if domains:
        return "run_questions_" + "_".join(domains)
    return "run_questions"


def _load_existing_results(path: Path) -> dict[str, dict]:
    """读取已完成结果，续跑时用 qid 去重。"""
    if not path.exists():
        return {}
    return {row["qid"]: row for row in read_jsonl(path) if row.get("qid")}


if __name__ == "__main__":
    main()
