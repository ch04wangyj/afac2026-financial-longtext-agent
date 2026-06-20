from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.retrieve.doc_first import (
    aggregate_chunk_hits_to_doc_ids,
    expand_top_doc_structural_neighbors,
    normalize_company_like_text,
    rank_document_candidates,
    retrieve_candidate_doc_ids,
    retrieve_doc_first,
    rerank_doc_first_chunks,
)
from agent.retrieve.probe import summarize_probe_results
from agent.schemas import Chunk, RetrievalResult

from agent.retrieve.probe_cases import PROBE_CASES_BY_NAME


BYD_CASE = PROBE_CASES_BY_NAME["byd_2025_net_profit"]
MIDEA_2024_CASE = PROBE_CASES_BY_NAME["midea_2024_share_repurchase"]
KEYWORD_BUNDLES = BYD_CASE["keyword_bundles"]
TARGET_ANSWER_TERMS = BYD_CASE["target_answer_terms"]
TARGET_DOC_ID = BYD_CASE["target_doc_id"]


def test_normalize_company_like_text_removes_generic_suffixes():
    assert normalize_company_like_text("比亚迪集团") == "比亚迪"
    assert normalize_company_like_text("美的集团股份有限公司") == "美的"


def test_normalize_company_like_text_keeps_non_company_terms():
    assert normalize_company_like_text("归属于母公司所有者的净利润") == "归属于母公司所有者的净利润"


def _chunk(doc_id: str, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:1",
        doc_id=doc_id,
        domain="financial_reports",
        page=1,
        section="table",
        clause_id="",
        text=text,
        metadata={"title": title},
    )


def _result(chunk_id: str, text: str, score: float, *, page: int = 1, doc_id: str = "annual_byd_2025_report") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_reports",
        score=score,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text=text,
        metadata={"page": page, "title": doc_id, "section": "table", "chunk_type": "table"},
    )


def test_rank_document_candidates_prefers_matching_company_year_and_metric():
    chunks = [
        _chunk("midea", "annual_midea_2025_report", "美的集团 2025 归属于母公司所有者的净利润"),
        _chunk("byd", "annual_byd_2025_report", "比亚迪集团 2025 归属于母公司所有者的净利润"),
        _chunk("byd_old", "annual_byd_2024_report", "比亚迪集团 2024 归属于母公司所有者的净利润"),
    ]

    ranked = rank_document_candidates(
        chunks,
        keyword_bundles=[("比亚迪集团", "2025", "归属于母公司所有者的净利润")],
        top_n=3,
    )

    assert ranked[0].doc_id == "byd"
    assert ranked[0].score > ranked[1].score


def test_summary_matches_answer_number_even_with_spacing_variation():
    results = [
        _result(
            "c1",
            "归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
            1.0,
        )
    ]
    summary = summarize_probe_results(
        method="doc_first",
        results=results,
        target_doc_id="annual_byd_2025_report",
        answer_terms=("归属于母公司所有者的净利润", "32,619,022"),
        top_k=5,
    )

    assert summary["answer_chunk_rank"] == 1


def test_rerank_prefers_income_statement_profit_block_over_employee_holding_block():
    employee_block = RetrievalResult(
        chunk_id="emp",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="比亚迪集团的中层管理人员、核心骨干员工 | 21,417 | 无 | 2025 年员工持股计划",
        metadata={"page": 73, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )
    profit_block = RetrievalResult(
        chunk_id="profit",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=0.6,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [employee_block, profit_block],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 10.0},
    )

    assert ranked[0].chunk_id == "profit"


def test_rerank_prefers_profit_block_over_cash_dividend_ratio_block():
    dividend_block = RetrievalResult(
        chunk_id="dividend",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="现金分红(含税) | 1,526.57 | 2,289.85 | 归属于母公司所有者的净利润 | 2,373.96 | 482.69 | 4,755.39 | 现金分红/归属于母公司所有者的净利润 | 68.30%",
        metadata={"page": 157, "title": "text05", "section": "table", "chunk_type": "table"},
    )
    profit_block = RetrievalResult(
        chunk_id="profit",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=0.7,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346 | 利润总额 | 39,753,049",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [dividend_block, profit_block],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"text05": 10.0, "annual_byd_2025_report": 8.0},
    )

    assert ranked[0].chunk_id == "profit"


def test_rerank_prefers_top_doc_candidate_when_structure_is_similar():
    other_doc = RetrievalResult(
        chunk_id="other",
        doc_id="annual_midea_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="归属于母公司股东的净利润 | 归属于母公司股东权益 | 2025 年 | 2024 年 | 43,945,411 | 38,537,237",
        metadata={"page": 10, "title": "annual_midea_2025_report", "section": "table", "chunk_type": "table"},
    )
    target_doc = RetrievalResult(
        chunk_id="target",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=0.7,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 | 营业收入 | 803,964,958 | 六、 | 按所有权归属分类 | 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [other_doc, target_doc],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 12.0, "annual_midea_2025_report": 8.0},
    )

    assert ranked[0].chunk_id == "target"


def test_expand_top_doc_structural_neighbors_adds_same_doc_same_page_answer_row():
    seed = _result(
        "seed",
        "加权平均 净资产收益率 | 归属于母公司普通股股东的净利润 | 3.58",
        1.0,
        page=267,
    )
    answer = _result(
        "answer",
        "附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        0.4,
        page=126,
    )
    expanded = expand_top_doc_structural_neighbors(
        [seed],
        candidate_neighbors=[answer],
        top_docs={"annual_byd_2025_report"},
    )

    assert any(item.chunk_id == "answer" for item in expanded)


def test_rerank_prefers_answer_chunk_when_same_section_has_dense_support():
    competitor = RetrievalResult(
        chunk_id="emp2",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="比亚迪集团的中层管理人员、核心骨干员工 | 21,417 | 无 | 2025 年员工持股计划",
        metadata={"page": 73, "title": "annual_byd_2025_report", "section": "员工持股计划", "chunk_type": "table"},
    )
    support = RetrievalResult(
        chunk_id="support",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=0.62,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 利润总额 | 39,753,049",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "附注七", "chunk_type": "table"},
    )
    answer = RetrievalResult(
        chunk_id="answer2",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=0.60,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 | 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "附注七", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [competitor, support, answer],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 10.0},
    )

    assert ranked[0].chunk_id == "answer2"


def test_doc_first_rerank_should_not_depend_on_bundle_term_hit_count():
    probe = RetrievalResult(
        chunk_id="p1",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [probe],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 10.0},
    )

    assert "term_hits" not in ranked[0].metadata.get("doc_first_rerank_features", {})


def test_doc_first_rerank_removes_metric_bonus_feature():
    probe = RetrievalResult(
        chunk_id="p2",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [probe],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 10.0},
    )

    features = ranked[0].metadata.get("doc_first_rerank_features", {})
    assert "metric_bonus" not in features


def test_doc_first_rerank_keeps_number_density_bonus_small_and_exposes_doc_cooccurrence_bonus():
    probe = RetrievalResult(
        chunk_id="p3",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="附注七 | 2025 年 | 2024 年 六、 | 按所有权归属分类 归属于母公司所有者的净利润 | 32,619 , 022 | 40,254,346",
        metadata={"page": 126, "title": "annual_byd_2025_report", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [probe],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"annual_byd_2025_report": 10.0},
    )

    features = ranked[0].metadata.get("doc_first_rerank_features", {})
    assert features["number_density_bonus"] <= 0.05
    assert "doc_keyword_cooccurrence_bonus" in features


def test_doc_first_rerank_softens_distractor_penalties():
    dividend_block = RetrievalResult(
        chunk_id="dividend2",
        doc_id="annual_byd_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="比亚迪集团 2025 归属于母公司所有者的净利润",
        evidence_text="现金分红(含税) | 1,526.57 | 2,289.85 | 归属于母公司所有者的净利润 | 2,373.96 | 482.69 | 4,755.39 | 现金分红/归属于母公司所有者的净利润 | 68.30%",
        metadata={"page": 157, "title": "text05", "section": "table", "chunk_type": "table"},
    )

    ranked = rerank_doc_first_chunks(
        [dividend_block],
        keyword_bundles=KEYWORD_BUNDLES,
        doc_scores={"text05": 10.0},
    )

    features = ranked[0].metadata.get("doc_first_rerank_features", {})
    assert features["distractor_penalty"] <= 0.25
    assert features["ratio_distractor_penalty"] <= 0.30


def test_aggregate_chunk_hits_to_doc_ids_rewards_keyword_cooccurrence_mildly():
    chunks = [
        _chunk("target", "annual_byd_2025_report", "比亚迪集团 2025 年 年度报告"),
        _chunk("target", "annual_byd_2025_report", "归属于母公司所有者的净利润 32,619,022"),
        _chunk("other", "annual_other_2025_report", "比亚迪集团 2025 年"),
        _chunk("other", "annual_other_2025_report", "unrelated text"),
    ]
    index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")

    ranked = aggregate_chunk_hits_to_doc_ids(
        index,
        keyword_bundles=[("比亚迪集团", "2025", "归属于母公司所有者的净利润")],
        top_docs=2,
        per_query_top_k=10,
        scoring_mode="bm25f_lite",
    )

    assert ranked[0] == "target"


def test_doc_first_prefers_target_narrative_repurchase_doc_over_newer_incentive_doc():
    chunks = [
        _chunk(
            "annual_midea_2024_report",
            "annual_midea_2024_report",
            "在稳定分红派现的同时，公司持续推出实施了一系列股份回购的方案，自2019年起公司连续四年推出回购计划，持续用于实施公司股权激励计划及员工持股计划，维护公司市值稳定与全体股东利益。",
        ),
        _chunk(
            "annual_midea_2024_report",
            "annual_midea_2024_report",
            "四、股份回购在报告期的具体实施情况 公司持续实施股份回购计划，并与员工持股计划及股权激励安排协同推进。",
        ),
        _chunk(
            "annual_midea_2025_report",
            "annual_midea_2025_report",
            "公司于2025年召开董事会，审议通过关于限制性股票激励计划部分激励股份回购注销的议案。",
        ),
        _chunk(
            "annual_midea_2025_report",
            "annual_midea_2025_report",
            "2025年公司继续实施限制性股票激励计划，涉及回购注销、解除限售及激励对象调整。",
        ),
    ]
    index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")

    results = retrieve_doc_first(
        index,
        keyword_bundles=[
            ("美的", "2019", "股份回购方案"),
            ("美的集团", "2019", "连续实施", "股份回购方案"),
            ("美的集团", "2019", "实施", "股份回购方案"),
        ],
        top_docs=2,
        top_k=10,
    )

    assert results[0].doc_id == "annual_midea_2024_report"


def test_aggregate_chunk_hits_can_use_offline_expansion_style_fields_for_midea_repurchase():
    chunks = [
        Chunk(
            chunk_id="midea-target",
            doc_id="annual_midea_2024_report",
            domain="financial_reports",
            page=1,
            section="股东回报",
            clause_id="",
            text="公司自2019年起连续实施股份回购计划，用于股权激励及员工持股计划。",
            tables=[],
            numbers=["2019 年"],
            dates=[],
            metadata={
                "title": "annual_midea_2024_report",
                "extra_index_fields": [
                    "美的集团",
                    "股份回购",
                    "2019 年",
                    "结论重述: 美的 2019 年 连续实施股份回购方案",
                ],
            },
        ),
        Chunk(
            chunk_id="midea-other",
            doc_id="annual_midea_2025_report",
            domain="financial_reports",
            page=1,
            section="股权激励",
            clause_id="",
            text="公司继续推进限制性股票激励计划并进行回购注销。",
            tables=[],
            numbers=["2025 年"],
            dates=[],
            metadata={"title": "annual_midea_2025_report", "extra_index_fields": ["股权激励", "回购注销"]},
        ),
    ]
    index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")

    ranked = aggregate_chunk_hits_to_doc_ids(
        index,
        keyword_bundles=MIDEA_2024_CASE["keyword_bundles"],
        top_docs=2,
        per_query_top_k=10,
        scoring_mode="bm25f_lite",
    )

    assert ranked[0] == "annual_midea_2024_report"


def test_doc_first_retrieval_should_surface_midea_2024_repurchase_doc_and_answer_chunk(bm25_index):
    results = retrieve_doc_first(
        bm25_index,
        keyword_bundles=MIDEA_2024_CASE["keyword_bundles"],
        top_docs=12,
        top_k=30,
    )
    summary = summarize_probe_results(
        method="doc_first_midea_2024",
        results=results,
        target_doc_id=MIDEA_2024_CASE["target_doc_id"],
        answer_terms=MIDEA_2024_CASE["target_answer_terms"],
        answer_match_mode=MIDEA_2024_CASE.get("answer_match_mode", "all_terms_in_one_chunk"),
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary


def test_doc_first_retrieval_should_surface_midea_2024_repurchase_doc_and_answer_chunk(bm25_index):
    results = retrieve_doc_first(
        bm25_index,
        keyword_bundles=MIDEA_2024_CASE["keyword_bundles"],
        top_docs=12,
        top_k=30,
    )
    summary = summarize_probe_results(
        method="doc_first_midea_2024",
        results=results,
        target_doc_id=MIDEA_2024_CASE["target_doc_id"],
        answer_terms=MIDEA_2024_CASE["target_answer_terms"],
        answer_match_mode=MIDEA_2024_CASE.get("answer_match_mode", "all_terms_in_one_chunk"),
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary


def test_doc_first_rerank_prefers_year_consistent_narrative_action_chunk():
    target = RetrievalResult(
        chunk_id="m24",
        doc_id="annual_midea_2024_report",
        domain="financial_reports",
        score=0.8,
        source="test",
        query="美的集团 2019 连续实施 股份回购方案",
        evidence_text="在稳定分红派现的同时，公司持续推出实施了一系列股份回购的方案，自2019年起公司连续四年推出回购计划，持续用于实施公司股权激励计划及员工持股计划，维护公司市值稳定与全体股东利益。",
        metadata={"page": 88, "title": "annual_midea_2024_report", "section": "", "chunk_type": "text"},
    )
    distractor = RetrievalResult(
        chunk_id="m25",
        doc_id="annual_midea_2025_report",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="美的集团 2019 连续实施 股份回购方案",
        evidence_text="公司于2025年召开董事会，审议通过关于限制性股票激励计划部分激励股份回购注销的议案。",
        metadata={"page": 82, "title": "annual_midea_2025_report", "section": "", "chunk_type": "text"},
    )

    ranked = rerank_doc_first_chunks(
        [distractor, target],
        keyword_bundles=MIDEA_2024_CASE["keyword_bundles"],
        doc_scores={"annual_midea_2025_report": 12.0, "annual_midea_2024_report": 11.0},
    )

    assert ranked[0].doc_id == "annual_midea_2024_report"


def test_retrieve_candidate_doc_ids_uses_document_index():
    chunks = [
        _chunk("byd", "annual_byd_2025_report", "比亚迪集团 2025 年 年度报告 归属于母公司所有者的净利润"),
        _chunk("midea", "annual_midea_2025_report", "美的集团 2025 年 年度报告 归属于母公司所有者的净利润"),
        _chunk("byd_old", "annual_byd_2024_report", "比亚迪集团 2024 年 年度报告 归属于母公司所有者的净利润"),
    ]
    doc_index = DocumentSearchIndex(BM25SearchIndex.build(chunks, tokenizer_mode="mixed"))

    ranked = retrieve_candidate_doc_ids(
        doc_index,
        keyword_bundles=[("比亚迪集团", "2025", "归属于母公司所有者的净利润")],
        top_docs=3,
    )

    assert ranked[0] == "byd"


def test_retrieve_candidate_doc_ids_keeps_target_doc_in_broad_shortlist(bm25_index):
    doc_index = DocumentSearchIndex.build(bm25_index.chunks, tokenizer_mode=bm25_index.tokenizer_mode)
    ranked = retrieve_candidate_doc_ids(
        doc_index,
        keyword_bundles=KEYWORD_BUNDLES,
        top_docs=12,
    )

    assert TARGET_DOC_ID in ranked


def test_aggregate_chunk_hits_to_doc_ids_can_rank_target_doc_first(bm25_index):
    ranked = aggregate_chunk_hits_to_doc_ids(
        bm25_index,
        keyword_bundles=KEYWORD_BUNDLES,
        top_docs=12,
        per_query_top_k=80,
        scoring_mode="bm25f_lite",
    )

    assert ranked[0] == TARGET_DOC_ID


def test_doc_first_retrieval_finds_byd_2025_document_and_answer_chunk(bm25_index):
    results = retrieve_doc_first(
        bm25_index,
        keyword_bundles=KEYWORD_BUNDLES,
        top_docs=12,
        top_k=30,
    )
    summary = summarize_probe_results(
        method="doc_first",
        results=results,
        target_doc_id=TARGET_DOC_ID,
        answer_terms=TARGET_ANSWER_TERMS,
        top_k=10,
    )

    assert summary["target_doc_in_top_5"], summary
    assert summary["answer_chunk_in_top_10"], summary
