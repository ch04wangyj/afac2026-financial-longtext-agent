"""脚本 07：分层抽样运行 A 榜题目。

用于在真实 API 成本可控的前提下覆盖不同领域和题型，并记录抽样清单。
"""

from __future__ import annotations

import argparse
import os
import random
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
from agent.io.jsonl import read_jsonl, write_json, write_jsonl
from agent.io.output_layout import choose_output_dir
from agent.llm.qwen_client import QwenClient
from agent.reasoning.solver import Solver
from agent.retrieve.retriever import Retriever


def main() -> None:
    """按领域和题型抽样，运行完整求解链路。"""
    parser = argparse.ArgumentParser(description="Run a stratified sample of A-board questions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Qwen.")
    parser.add_argument("--a-board-quality", action="store_true", help="Enable A-board LogicRAG retrieval mode with doc-scoped coverage gate and financial calculator.")
    parser.add_argument("--sample-size", type=int, default=20, help="Total number of questions.")
    parser.add_argument("--per-domain", type=int, default=4, help="Target number of questions per domain.")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--qids", nargs="*", default=None, help="Explicit qids to run instead of sampling.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing answer_results.jsonl.")
    args = parser.parse_args()

    if args.a_board_quality:
        _enable_a_board_quality_mode()

    settings = Settings.from_env()
    settings.ensure_dirs()
    rng = random.Random(args.seed)
    questions = load_questions(settings.questions_root, domains=args.domains)
    sample = select_by_qids(questions, args.qids) if args.qids else select_stratified_sample(
        questions, args.sample_size, args.per_domain, rng
    )
    run_scope = "a100" if len(sample) == 100 else "sample"
    run_name = "full100" if run_scope == "a100" else f"sample{len(sample)}"
    run_dir = choose_output_dir(
        settings,
        run_scope=run_scope,
        run_name=run_name,
        strategy=settings.retrieval_strategy,
        dry_run=args.dry_run,
        resume=args.resume,
    )

    manifest = [
        {
            "qid": question.qid,
            "domain": question.domain,
            "answer_format": question.answer_format,
            "type": question.type,
            "doc_ids": question.doc_ids,
        }
        for question in sample
    ]
    write_json(run_dir / "sample_manifest.json", manifest)

    solver = _build_solver(settings, dry_run=args.dry_run)
    out_path = run_dir / "answer_results.jsonl"
    existing_by_qid = _load_existing_results(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    rows = []
    todo_questions = [question for question in sample if question.qid not in existing_by_qid]
    if existing_by_qid:
        for idx, question in enumerate(sample, start=1):
            if question.qid in existing_by_qid:
                print(
                    f"[{idx}/{len(sample)}] skip {question.domain}/{question.qid}/{question.answer_format} from checkpoint",
                    flush=True,
                )
                rows.append(existing_by_qid[question.qid])

    def solve_question(question):
        print(f"[{question.domain}/{question.qid}/{question.answer_format}] solving", flush=True)
        return question.qid, solver.solve(question).to_dict()

    with ThreadPoolExecutor(max_workers=settings.question_workers) as executor:
        futures = {executor.submit(solve_question, question): question.qid for question in todo_questions}
        pending = []
        for future in as_completed(futures):
            pending.append(future.result())

    solved_by_qid = {qid: row for qid, row in pending}
    rows.extend(solved_by_qid[question.qid] for question in todo_questions)
    write_jsonl(out_path, rows)
    print(f"wrote {len(rows)} sampled results -> {run_dir}", flush=True)



def _enable_a_board_quality_mode() -> None:
    """通过环境变量打开 A 榜 LogicRAG 检索优化模式。"""
    os.environ["AFAC_LOGICRAG_ENABLED"] = "true"
    os.environ["AFAC_RETRIEVAL_STRATEGY"] = "logicrag_agent"
    os.environ["AFAC_A_BOARD_OPTION_MATRIX_ENABLED"] = "false"
    os.environ["AFAC_ENABLE_MULTI_OPTION_JUDGEMENT"] = "false"
    os.environ["AFAC_A_BOARD_COVERAGE_GATE_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED"] = "true"
    os.environ["AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY"] = "false"



def select_stratified_sample(questions, sample_size: int, per_domain: int, rng: random.Random):

    by_domain: dict[str, list] = defaultdict(list)
    for question in questions:
        by_domain[question.domain].append(question)

    selected = []
    seen: set[str] = set()
    for domain in sorted(by_domain):
        domain_questions = by_domain[domain][:]
        rng.shuffle(domain_questions)
        selected.extend(_pick_for_domain(domain_questions, per_domain, seen))

    remaining = [question for question in questions if question.qid not in seen]
    rng.shuffle(remaining)
    for question in remaining:
        if len(selected) >= sample_size:
            break
        selected.append(question)
        seen.add(question.qid)
    return selected[:sample_size]


def select_by_qids(questions, qids: list[str]):
    """按用户指定 qid 保持顺序选题。"""
    by_qid = {question.qid: question for question in questions}
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise KeyError(f"Unknown qids: {missing}")
    return [by_qid[qid] for qid in qids]


def _pick_for_domain(questions, per_domain: int, seen: set[str]):
    """单领域内优先覆盖三种 answer_format，再随机补足。"""
    picked = []
    for answer_format in ("multi", "mcq", "tf"):
        for question in questions:
            if question.answer_format == answer_format and question.qid not in seen:
                picked.append(question)
                seen.add(question.qid)
                break
        if len(picked) >= per_domain:
            return picked
    for question in questions:
        if len(picked) >= per_domain:
            break
        if question.qid in seen:
            continue
        picked.append(question)
        seen.add(question.qid)
    return picked


def _build_solver(settings: Settings, dry_run: bool) -> Solver:
    """构建和 03_run_questions.py 一致的 Solver。"""
    index = BM25SearchIndex.load(settings.index_dir / "bm25_index.pkl")
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
    return Solver(retriever, compressor, QwenClient(settings, dry_run=dry_run))


def _load_existing_results(path: Path) -> dict[str, dict]:
    """读取已完成结果，续跑时用 qid 去重。"""
    if not path.exists():
        return {}
    return {row["qid"]: row for row in read_jsonl(path) if row.get("qid")}


if __name__ == "__main__":
    main()
