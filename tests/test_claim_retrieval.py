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
