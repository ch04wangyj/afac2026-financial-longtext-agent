"""脚本 05：横向比较不同 RAG/tokenizer 检索方案。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.data.questions import load_questions
from agent.index.bm25 import BM25SearchIndex
from agent.index.build import load_chunks
from agent.io.jsonl import write_json
from agent.retrieve.metrics import evaluate_retrieval, summarize_metrics
from agent.retrieve.variants import RAG_VARIANTS, retrieve_with_variant


def main() -> None:
    """用 A 组 doc_ids 作为代理标签评估检索命中。"""
    parser = argparse.ArgumentParser(
        description="Compare retrieval-only RAG variants using A-board doc_ids as gold document labels."
    )
    parser.add_argument("--chunks", type=Path, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--tokenizer-modes", nargs="*", default=["mixed", "char", "word"], choices=["mixed", "char", "word"])
    parser.add_argument("--variants", nargs="*", default=[variant.name for variant in RAG_VARIANTS])
    parser.add_argument("--include-missing-gold", action="store_true", help="Evaluate questions even if indexed chunks miss gold docs.")
    parser.add_argument("--output-name", default="rag_compare", help="Subdirectory under outputs/ for reports.")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    chunks_path = args.chunks or settings.processed_dir / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    indexed_doc_ids = {chunk.doc_id for chunk in chunks}
    questions = load_questions(settings.questions_root, domains=args.domains)
    if not args.include_missing_gold:
        questions = [q for q in questions if set(q.doc_ids).issubset(indexed_doc_ids)]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("No questions to evaluate. Run 01_prepare_docs.py for the target documents first.")

    detail_rows = []
    metric_rows = []
    for tokenizer_mode in args.tokenizer_modes:
        # 每个 tokenizer 单独建内存索引，避免不同分词模式互相污染。
        print(f"building in-memory index tokenizer={tokenizer_mode} chunks={len(chunks)}")
        index = BM25SearchIndex.build(chunks, tokenizer_mode=tokenizer_mode)
        for variant_name in args.variants:
            print(f"evaluating tokenizer={tokenizer_mode} variant={variant_name} questions={len(questions)}")
            for question in questions:
                results = retrieve_with_variant(index, question, variant_name, top_k=args.top_k)
                metric = evaluate_retrieval(question, results, variant_name, tokenizer_mode)
                metric_rows.append(metric)
                detail_rows.append(metric.to_dict())

    summary_rows = summarize_metrics(metric_rows)
    out_dir = settings.outputs_dir / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "detail.csv", detail_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    write_json(out_dir / "detail.json", detail_rows)
    write_json(out_dir / "summary.json", summary_rows)
    _write_markdown(out_dir / "report.md", summary_rows)
    print(f"wrote RAG comparison report -> {out_dir}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    """写 CSV 报告，使用 utf-8-sig 方便 Excel 直接打开。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    """写 Markdown 汇总表，默认只展示 ALL 聚合行。"""
    all_rows = [row for row in rows if row["domain"] == "ALL"]
    all_rows = sorted(all_rows, key=lambda row: (row["recall_at_10"], row["hit_at_5"], row["mrr_at_10"]), reverse=True)
    lines = [
        "# RAG Variant Comparison",
        "",
        "A榜没有公开标准答案，因此本报告先用题目给定的 `doc_ids` 作为检索命中代理指标。",
        "`recall_at_10` 越高，说明越能把相关文档送入后续推理；不等于最终答题准确率。",
        "",
        "| rank | tokenizer | variant | questions | hit@1 | hit@5 | hit@10 | recall@10 | all_gold@10 | mrr@10 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(all_rows, start=1):
        lines.append(
            "| {rank} | {tokenizer_mode} | {variant} | {questions} | {hit_at_1:.3f} | {hit_at_5:.3f} | "
            "{hit_at_10:.3f} | {recall_at_10:.3f} | {all_gold_at_10:.3f} | {mrr_at_10:.3f} |".format(
                rank=rank,
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
