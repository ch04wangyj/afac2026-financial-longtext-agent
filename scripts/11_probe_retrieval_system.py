"""Probe retrieval methods against fixed keyword bundles.

This script is diagnostic only. It does not call an LLM and does not mutate indexes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.doc_first import retrieve_doc_first
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.probe import summarize_probe_results

from agent.retrieve.probe_cases import PROBE_CASES


TARGETS = PROBE_CASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe retrieval methods on the BYD 2025 keyword benchmark.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    index = BM25SearchIndex.load(settings.index_dir / "bm25_index.pkl")

    summaries = []
    for case in TARGETS:
        for method, results in _run_methods(index, case, top_k=args.top_k).items():
            summaries.append(
                summarize_probe_results(
                    method=f"{case['name']}::{method}",
                    results=results,
                    target_doc_id=case["target_doc_id"],
                    answer_terms=case["target_answer_terms"],
                    top_k=10,
                )
            )

    payload = {"cases": TARGETS, "summaries": summaries}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


def _run_methods(index: BM25SearchIndex, case: dict, *, top_k: int) -> dict[str, list]:
    keyword_bundles = case["keyword_bundles"]
    first_query = _bundle_query(keyword_bundles[0])
    return {
        "plain_bm25_first_bundle": index.search(first_query, top_k=top_k, source="plain_bm25_first_bundle"),
        "bm25f_lite_first_bundle": index.search(
            first_query,
            top_k=top_k,
            source="bm25f_lite_first_bundle",
            scoring_mode="bm25f_lite",
        ),
        "bundle_rrf_bm25": _bundle_rrf(index, keyword_bundles, top_k=top_k, scoring_mode=None),
        "bundle_rrf_bm25f_lite": _bundle_rrf(index, keyword_bundles, top_k=top_k, scoring_mode="bm25f_lite"),
        "doc_first_chunk_rerank": retrieve_doc_first(
            index,
            keyword_bundles=keyword_bundles,
            top_docs=12,
            top_k=top_k,
        ),
    }


def _bundle_rrf(index: BM25SearchIndex, keyword_bundles, *, top_k: int, scoring_mode: str | None):
    ranked_lists = [
        index.search(
            _bundle_query(bundle),
            top_k=top_k,
            source="bundle_rrf",
            scoring_mode=scoring_mode,
        )
        for bundle in keyword_bundles
    ]
    return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


def _bundle_query(bundle: tuple[str, ...]) -> str:
    return " ".join(bundle)


if __name__ == "__main__":
    main()
