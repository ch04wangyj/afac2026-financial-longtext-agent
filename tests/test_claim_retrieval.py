from agent.retrieve.claims import build_claim_targets
from agent.schemas import Question, RetrievalResult



def test_build_claim_query_bundles_returns_intent_labeled_queries():
    from agent.retrieve.claim_retrieval import build_claim_query_bundles

    question = Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="比较公司2025年营业收入和净利润变化。",
        options={"A": "营业收入同比增长", "B": "净利润同比下降"},
        answer_format="multi",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    bundles = build_claim_query_bundles(question, claim, max_bundles=5)

    assert bundles
    assert all(bundle.query for bundle in bundles)
    assert all(bundle.intent for bundle in bundles)
    assert {bundle.intent for bundle in bundles} >= {"stem_claim", "entity_anchor"}
    assert any(bundle.intent in {"metric_value", "comparison_endpoint"} for bundle in bundles)



def test_build_claim_query_bundles_deduplicates_queries():
    from agent.retrieve.claim_retrieval import build_claim_query_bundles

    question = Question(
        qid="q2",
        domain="regulatory",
        split="A",
        question="证券公司被行政处罚是否会扣减分类评价得分？",
        options={"A": "会扣减分类评价得分"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    bundles = build_claim_query_bundles(question, claim, max_bundles=8)
    queries = [bundle.query for bundle in bundles]

    assert len(queries) == len(set(queries))
    assert any(bundle.intent == "clause_consequence" for bundle in bundles)


def test_financial_metric_alias_query_expands_report_disclosure_names():
    from agent.retrieve.claim_retrieval import build_claim_query_bundles

    question = Question(
        qid="q_alias",
        domain="financial_reports",
        split="A",
        question="比较两家公司2025年经营成果。",
        options={"A": "甲公司归母净利润增速高于乙公司"},
        answer_format="mcq",
        doc_ids=["doc_a", "doc_b"],
    )

    claim = build_claim_targets(question)[0]
    bundles = build_claim_query_bundles(question, claim, max_bundles=6)
    alias_queries = [bundle.query for bundle in bundles if bundle.intent == "metric_alias"]

    assert len(alias_queries) == 1
    assert "归属于上市公司股东的净利润" in alias_queries[0]
    assert "本年比上年增减" in alias_queries[0]
    assert "净利润" in claim.must_terms[0]


def test_dividend_alias_query_covers_h_share_final_dividend_terms():
    from agent.retrieve.claim_retrieval import build_claim_query_bundles

    question = Question(
        qid="q_dividend_alias",
        domain="financial_reports",
        split="A",
        question="比较两家公司每股现金分红金额。",
        options={"A": "甲公司的每股现金分红高于乙公司"},
        answer_format="mcq",
        doc_ids=["doc_a", "doc_b"],
    )

    claim = build_claim_targets(question)[0]
    bundles = build_claim_query_bundles(question, claim, max_bundles=8)
    alias_query = next(bundle.query for bundle in bundles if bundle.intent == "metric_alias")
    focused_query = next(bundle.query for bundle in bundles if bundle.intent == "dividend_final")

    assert "末期股息" in alias_query
    assert "建议派发末期股息" in alias_query
    assert "每股股息" in alias_query
    assert focused_query.startswith("末期股息 建议派发 每10股")



def test_retrieve_claim_candidates_uses_intent_sources_and_doc_filter():
    from agent.retrieve.claim_retrieval import retrieve_claim_candidates

    question = Question(
        qid="q3",
        domain="regulatory",
        split="A",
        question="证券公司被行政处罚是否会扣减分类评价得分？",
        options={"A": "会扣减分类评价得分"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    class FakeIndex:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k, filter_doc_ids=None, source="test"):
            self.calls.append((query, top_k, filter_doc_ids, source))
            return [
                RetrievalResult(
                    chunk_id=f"{source}:1",
                    doc_id="doc1",
                    domain="regulatory",
                    score=1.0,
                    source=source,
                    query=query,
                    evidence_text=query,
                    metadata={},
                )
            ]

    index = FakeIndex()
    results, bundles = retrieve_claim_candidates(
        index,
        question,
        claim,
        filter_doc_ids={"doc1"},
        top_k_per_query=3,
        fused_top_k=5,
    )

    assert results
    assert bundles
    assert all(call[2] == {"doc1"} for call in index.calls)
    assert any(call[3].startswith("claim_A_") for call in index.calls)


def test_comparison_claim_searches_each_document_independently():
    from agent.retrieve.claim_retrieval import retrieve_claim_candidates

    question = Question(
        qid="q4",
        domain="financial_reports",
        split="A",
        question="比较两家公司2025年净利润增速。",
        options={"A": "甲公司净利润增速高于乙公司"},
        answer_format="mcq",
        doc_ids=["doc_a", "doc_b"],
    )
    claim = build_claim_targets(question)[0]

    class FakeIndex:
        def __init__(self):
            self.filters = []

        def search(self, query, top_k, filter_doc_ids=None, source="test"):
            self.filters.append(filter_doc_ids)
            doc_id = sorted(filter_doc_ids or {"doc_a"})[0]
            return [
                RetrievalResult(
                    chunk_id=f"{doc_id}:{source}",
                    doc_id=doc_id,
                    domain="financial_reports",
                    score=1.0,
                    source=source,
                    query=query,
                    evidence_text=query,
                    metadata={},
                )
            ]

    index = FakeIndex()
    retrieve_claim_candidates(
        index,
        question,
        claim,
        filter_doc_ids={"doc_a", "doc_b"},
        top_k_per_query=3,
        fused_top_k=8,
    )

    assert {"doc_a"} in index.filters
    assert {"doc_b"} in index.filters
