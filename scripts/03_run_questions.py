"""脚本 03：运行检索 + 压缩 + Qwen 作答闭环。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.config import Settings
from agent.data.questions import load_questions
from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.io.jsonl import append_jsonl, read_jsonl, write_jsonl
from agent.llm.qwen_client import QwenClient
from agent.reasoning.solver import Solver
from agent.retrieve.retriever import Retriever


def main() -> None:
    """执行题目求解；真实调用前建议用 --limit 控制规模。"""
    parser = argparse.ArgumentParser(description="Run retrieval + Qwen answering for A group questions.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call Qwen; emit deterministic dummy answers.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions for smoke tests.")
    parser.add_argument("--domains", nargs="*", default=None, help="Optional domains to include.")
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing answer_results.jsonl.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = load_questions(settings.questions_root, domains=args.domains)
    if args.limit:
        questions = questions[: args.limit]

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

    out_path = settings.outputs_dir / "answer_results.jsonl"
    existing_by_qid = _load_existing_results(out_path) if args.resume else {}
    if not args.resume and out_path.exists():
        out_path.unlink()

    results = []
    for idx, question in enumerate(questions, start=1):
        if question.qid in existing_by_qid:
            print(f"[{idx}/{len(questions)}] skip {question.qid} from checkpoint", flush=True)
            results.append(existing_by_qid[question.qid])
            continue

        # 每题即时打印进度，真实 API 调用时便于观察是否卡住。
        print(f"[{idx}/{len(questions)}] solving {question.qid}", flush=True)
        row = solver.solve(question).to_dict()
        results.append(row)
        append_jsonl(out_path, row)

    # 最终重写一次，保证 JSONL 顺序和本次题目顺序一致。
    write_jsonl(out_path, results)
    print(f"wrote {len(results)} answer results -> {out_path}", flush=True)


def _load_existing_results(path: Path) -> dict[str, dict]:
    """读取已完成结果，续跑时用 qid 去重。"""
    if not path.exists():
        return {}
    return {row["qid"]: row for row in read_jsonl(path) if row.get("qid")}


if __name__ == "__main__":
    main()
