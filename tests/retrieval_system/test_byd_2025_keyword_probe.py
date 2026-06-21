from __future__ import annotations

import pytest

from agent.retrieve.doc_first import retrieve_doc_first
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.probe import summarize_probe_results

from agent.retrieve.probe_cases import PROBE_CASES_BY_NAME


CASE = PROBE_CASES_BY_NAME["byd_2025_net_profit"]
KEYWORD_BUNDLES = CASE["keyword_bundles"]
TARGET_ANSWER_TERMS = CASE["target_answer_terms"]
TARGET_DOC_ID = CASE["target_doc_id"]


def _bundle_query(bundle: tuple[str, ...]) -> str:
    return " ".join(bundle)


def _plain_bm25(index, top_k: int = 30):
    return index.search(_bundle_query(KEYWORD_BUNDLES[0]), top_k=top_k, source="plain_bm25")


def _bm25f_lite(index, top_k: int = 30):
    return index.search(
        _bundle_query(KEYWORD_BUNDLES[0]),
        top_k=top_k,
        source="bm25f_lite",
        scoring_mode="bm25f_lite",
    )


def _bundle_rrf(index, *, scoring_mode: str | None = None, top_k: int = 30):
    ranked_lists = [
        index.search(
            _bundle_query(bundle),
            top_k=top_k,
            source="bundle_rrf",
            scoring_mode=scoring_mode,
        )
        for bundle in KEYWORD_BUNDLES
    ]
    return reciprocal_rank_fusion(ranked_lists, top_k=top_k)


@pytest.mark.parametrize(
    ("method", "runner"),
    [
        ("plain_bm25", _plain_bm25),
        ("bm25f_lite", _bm25f_lite),
        ("bundle_rrf", _bundle_rrf),
    ],
)
def test_baseline_keyword_probe_records_current_ranking(bm25_index, method, runner):
    results = runner(bm25_index, top_k=30)
    summary = summarize_probe_results(
        method=method,
        results=results,
        target_doc_id=TARGET_DOC_ID,
        answer_terms=TARGET_ANSWER_TERMS,
        top_k=10,
    )

    assert summary["top_results"], f"{method} returned no results"
    assert summary["doc_rank"] is not None, summary


@pytest.mark.xfail(reason="Current retrieval is expected to fail or be unstable for this simple keyword benchmark.")
def test_current_baseline_should_rank_answer_chunk_in_top_10(bm25_index):
    results = _bundle_rrf(bm25_index, top_k=30)
    summary = summarize_probe_results(
        method="bundle_rrf",
        results=results,
        target_doc_id=TARGET_DOC_ID,
        answer_terms=TARGET_ANSWER_TERMS,
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary


def test_doc_first_chunk_rerank_ranks_target_doc_and_answer_chunk(bm25_index):
    results = retrieve_doc_first(
        bm25_index,
        keyword_bundles=KEYWORD_BUNDLES,
        top_docs=12,
        top_k=30,
    )
    summary = summarize_probe_results(
        method="doc_first_chunk_rerank",
        results=results,
        target_doc_id=TARGET_DOC_ID,
        answer_terms=TARGET_ANSWER_TERMS,
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary
