"""脚本 07：分层抽样运行 A 榜题目。

用于在真实 API 成本可控的前提下覆盖不同领域和题型，并记录抽样清单。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.data.questions import load_questions
from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.io.jsonl import append_jsonl, read_jsonl, write_json, write_jsonl
from agent.llm.qwen_client import QwenClient
from agent.reasoning.solver import Solver
from agent.retrieve.retriever import Retriever


def main() -> None:
    """按领域和题型抽样，运行完整求解链路。"""
    parser = argparse.ArgumentParser(description="Run a stratified sample of A-board questions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Qwen.")
    parser.add_argument("--sample-size", type=int, default=20, help="Total number of questions.")
    parser.add_argument("--per-domain", type=int, default=4, help="Target number of questions per domain.")
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--qids", nargs="*", default=None, help="Explicit qids to run instead of sampling.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing answer_results.jsonl.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    rng = random.Random(args.seed)
    questions = load_questions(settings.questions_root, domains=args.domains)
    sample = select_by_qids(questions, args.qids) if args.qids else select_stratified_sample(
        questions, args.sample_size, args.per_domain, rng
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
    write_json(settings.outputs_dir / "sample_manifest.json", manifest)

    solver = _build_solver(settings, dry_run=args.dry_run)
    out_path = settings.outputs_dir / "answer_results.jsonl"
    existing_by_qid = _load_existing_results(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    rows = []
    for idx, question in enumerate(sample, start=1):
        if question.qid in existing_by_qid:
            print(
                f"[{idx}/{len(sample)}] skip {question.domain}/{question.qid}/{question.answer_format} from checkpoint",
                flush=True,
            )
            rows.append(existing_by_qid[question.qid])
            continue

        print(f"[{idx}/{len(sample)}] solving {question.domain}/{question.qid}/{question.answer_format}", flush=True)
        row = solver.solve(question).to_dict()
        rows.append(row)
        append_jsonl(out_path, row)

    write_jsonl(settings.outputs_dir / "answer_results.jsonl", rows)
    print(f"wrote {len(rows)} sampled results -> {settings.outputs_dir}", flush=True)


def select_stratified_sample(questions, sample_size: int, per_domain: int, rng: random.Random):
    """优先保证各领域覆盖，再尽量覆盖 mcq/multi/tf。"""
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
