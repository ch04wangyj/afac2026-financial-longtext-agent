from __future__ import annotations

import pytest

from agent.retrieve.doc_first import retrieve_doc_first
from agent.retrieve.probe import summarize_probe_results
from agent.retrieve.probe_cases import PROBE_CASES


PROBE_CASES_TO_CHECK = [
    pytest.param(PROBE_CASES[1], id=PROBE_CASES[1]["name"], marks=pytest.mark.xfail(reason="Current doc_first_chunk_rerank does not yet reliably surface the Midea 2025 target doc/chunk under this benchmark setup.")),
    pytest.param(PROBE_CASES[2], id=PROBE_CASES[2]["name"], marks=pytest.mark.xfail(reason="Current doc_first_chunk_rerank does not yet reliably surface the Midea 2024 target doc/chunk under this benchmark setup.")),
    pytest.param(PROBE_CASES[3], id=PROBE_CASES[3]["name"]),
]


@pytest.mark.parametrize("case", PROBE_CASES_TO_CHECK)
def test_doc_first_chunk_rerank_matches_added_probe_cases(bm25_index, case):
    results = retrieve_doc_first(
        bm25_index,
        keyword_bundles=case["keyword_bundles"],
        top_docs=12,
        top_k=30,
    )
    summary = summarize_probe_results(
        method=f"doc_first_chunk_rerank::{case['name']}",
        results=results,
        target_doc_id=case["target_doc_id"],
        answer_terms=case["target_answer_terms"],
        answer_match_mode=case.get("answer_match_mode", "all_terms_in_one_chunk"),
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary
