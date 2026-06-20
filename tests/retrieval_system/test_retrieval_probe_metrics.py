from agent.retrieve.probe import find_answer_chunk_rank, find_doc_rank, summarize_probe_results
from agent.schemas import RetrievalResult


def _result(rank: int, doc_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"c{rank}",
        doc_id=doc_id,
        domain="financial_reports",
        score=1.0 / rank,
        source="test",
        query="比亚迪集团 2025 归母净利润",
        evidence_text=text,
        metadata={"page": rank, "title": doc_id, "section": "table"},
    )


def test_find_doc_rank_returns_first_matching_rank():
    results = [_result(1, "other", "x"), _result(2, "annual_byd_2025_report", "x")]

    assert find_doc_rank(results, "annual_byd_2025_report") == 2


def test_find_answer_chunk_rank_requires_doc_and_all_terms():
    results = [
        _result(1, "annual_byd_2025_report", "归属于母公司所有者的净利润"),
        _result(2, "other", "归属于母公司所有者的净利润 32,619,022"),
        _result(3, "annual_byd_2025_report", "归属于母公司所有者的净利润 32,619,022"),
    ]

    assert find_answer_chunk_rank(
        results,
        target_doc_id="annual_byd_2025_report",
        answer_terms=("归属于母公司所有者的净利润", "32,619,022"),
    ) == 3


def test_find_answer_chunk_rank_allows_any_relevant_chunk_for_multi_chunk_cases():
    results = [
        _result(1, "annual_midea_2025_report", "营业总收入 458,502,407"),
        _result(2, "annual_midea_2025_report", "2025年，公司营业总收入4,585亿元，同比增长12%"),
        _result(3, "annual_midea_2025_report", "一、营业总收入\n458,502,407\n409,084,266"),
        _result(4, "annual_midea_2025_report", "12%"),
    ]

    assert find_answer_chunk_rank(
        results,
        target_doc_id="annual_midea_2025_report",
        answer_terms=(
            "2025年，公司营业总收入4,585亿元，同比增长12%",
            "12%",
            "一、营业总收入\n458,502,407\n409,084,266",
        ),
        answer_match_mode="any_term_in_target_doc_chunk",
    ) == 2


def test_summarize_probe_results_uses_any_relevant_chunk_rule_for_multi_chunk_cases():
    results = [
        _result(1, "annual_midea_2025_report", "2025年，公司营业总收入4,585亿元，同比增长12%"),
        _result(2, "annual_midea_2025_report", "一、营业总收入\n458,502,407\n409,084,266"),
        _result(3, "annual_midea_2025_report", "12%"),
    ]

    summary = summarize_probe_results(
        method="m",
        results=results,
        target_doc_id="annual_midea_2025_report",
        answer_terms=(
            "2025年，公司营业总收入4,585亿元，同比增长12%",
            "12%",
            "一、营业总收入\n458,502,407\n409,084,266",
        ),
        answer_match_mode="any_term_in_target_doc_chunk",
        top_k=3,
    )

    assert summary["doc_rank"] == 1
    assert summary["answer_chunk_rank"] == 1
    assert summary["answer_chunk_in_top_10"] is True
    assert "answer_coverage_rank" not in summary
    assert "answer_coverage_in_top_10" not in summary


def test_summarize_probe_results_includes_top_result_snippets():
    results = [_result(1, "annual_byd_2025_report", "归属于母公司所有者的净利润 32,619,022 abc")]

    summary = summarize_probe_results(
        method="m",
        results=results,
        target_doc_id="annual_byd_2025_report",
        answer_terms=("归属于母公司所有者的净利润", "32,619,022"),
        top_k=1,
    )

    assert summary["method"] == "m"
    assert summary["doc_rank"] == 1
    assert summary["answer_chunk_rank"] == 1
    assert summary["top_results"][0]["doc_id"] == "annual_byd_2025_report"
