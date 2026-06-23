"""脚本 24：运行 V15 精确验证器（PoT + 自验证 + 自适应路由 + LLM Judge）。

用法：
    python scripts/24_run_v15_verifier.py --output-dir outputs/v15_pot_verify --workers 8 --no-thinking
"""

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
from agent.reasoning.v15_verifier import V15PreciseVerifier, V15VerifierConfig
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V15 precise verifier with PoT + self-verification.")
    parser.add_argument("--qids", nargs="*", default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-context-chars", type=int, default=12_000)
    parser.add_argument("--strategy-name", default="v15_pot_verify")
    parser.add_argument("--disable-pot", action="store_true")
    parser.add_argument("--disable-self-verify", action="store_true")
    parser.add_argument("--disable-llm-judge", action="store_true")
    parser.add_argument("--disable-routing", action="store_true")
    args = parser.parse_args()

    os.environ["AFAC_QWEN_MODEL"] = args.model
    settings = Settings.from_env()
    settings.ensure_dirs()
    questions = _select_questions(
        load_questions(settings.questions_root, domains=args.domains), args.qids, args.limit
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.index or settings.processed_dir / "v14_layout" / "bm25_index.pkl"
    if not index_path.exists():
        print(f"ERROR: Index not found at {index_path}")
        sys.exit(1)

    config = V15VerifierConfig(
        strategy_name=args.strategy_name,
        max_context_chars=args.max_context_chars,
        enable_thinking=not args.no_thinking,
        enable_pot=not args.disable_pot,
        enable_self_verification=not args.disable_self_verify,
        enable_llm_judge=not args.disable_llm_judge,
        enable_adaptive_routing=not args.disable_routing,
    )

    verifier = V15PreciseVerifier(
        BM25SearchIndex.load(index_path),
        QwenClient(settings, dry_run=args.dry_run),
        config,
    )

    # Resume
    done_qids: set[str] = set()
    results_path = output_dir / "answer_results.jsonl"
    if args.resume and results_path.exists():
        for row in read_jsonl(results_path):
            done_qids.add(row["qid"])
        print(f"Resume: skipping {len(done_qids)} completed questions.")

    pending = [q for q in questions if q.qid not in done_qids]
    print(f"Running {len(pending)} questions with {args.workers} workers (strategy={args.strategy_name})")

    if not pending:
        print("All questions already completed.")
        _write_summary(output_dir, results_path)
        return

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    completed = 0

    def _solve(q):
        try:
            return verifier.solve(q)
        except Exception as exc:
            from agent.schemas import TokenUsage
            return AnswerResult(
                qid=q.qid,
                answer=sorted(q.options)[0],
                confidence=0.0,
                evidence=[],
                token_usage=TokenUsage(),
                raw_response="",
                metadata={"error": repr(exc), "domain": q.domain, "answer_format": q.answer_format},
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_solve, q): q for q in pending}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            row = result.to_dict()
            append_jsonl(results_path, row)
            u = result.token_usage
            total_usage["prompt_tokens"] += u.prompt_tokens
            total_usage["completion_tokens"] += u.completion_tokens
            total_usage["total_tokens"] += u.total_tokens
            routing = result.metadata.get("routing_strategy", "?")
            pot = result.metadata.get("pot_executed", False)
            sv = result.metadata.get("self_verified", None)
            print(
                f"[{completed}/{len(pending)}] {result.qid} ans={result.answer} conf={result.confidence:.2f} "
                f"route={routing} pot={pot} sv={sv} tok={u.total_tokens}",
                flush=True,
            )

    print(f"\nDone: {completed} questions, total_tokens={total_usage['total_tokens']:,}")
    _write_summary(output_dir, results_path)


def _write_summary(output_dir: Path, results_path: Path):
    results = list(read_jsonl(results_path))
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
    summary = {
        "total_questions": len(results),
        "total_tokens": total_tokens,
        "strategy": "v15_pot_verify",
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Summary: {len(results)} questions, {total_tokens:,} tokens -> {output_dir / 'summary.json'}")


def _select_questions(questions, qids, limit):
    if qids:
        qid_set = set(qids)
        questions = [q for q in questions if q.qid in qid_set]
    if limit:
        questions = questions[:limit]
    return questions


if __name__ == "__main__":
    main()
