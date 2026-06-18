from agent.schemas import Question, RetrievalResult

from agent.retrieve.option_retrieval import build_option_queries, retrieve_option_candidates


def _question() -> Question:
    return Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="根据两年年报，哪些说法正确？",
        options={"A": "2025 年营业收入增长", "B": "2025 年净利润下降"},
        answer_format="multi",
        type="财务指标对比分析",
        doc_ids=["d1", "d2"],
    )


def test_build_option_queries_keeps_each_option_separate():
    queries = build_option_queries(_question())

    assert set(queries) == {"A", "B"}
    assert any("营业收入" in query for query in queries["A"])
    assert any("净利润" in query for query in queries["B"])
    assert all("净利润下降" not in query for query in queries["A"])


def test_retrieve_option_candidates_returns_candidates_by_option():
    class FakeIndex:
        def search(self, query, top_k, filter_doc_ids=None, source=""):
            key = "A" if "收入" in query else "B"
            return [
                RetrievalResult(
                    chunk_id=f"{key}:1",
                    doc_id="d1",
                    domain="financial_reports",
                    score=1.0,
                    source=source,
                    query=query,
                    evidence_text=query,
                    metadata={},
                )
            ]

    candidates = retrieve_option_candidates(FakeIndex(), _question(), filter_doc_ids={"d1"}, top_k_per_query=3, fused_top_k=5)

    assert set(candidates) == {"A", "B"}
    assert candidates["A"][0].chunk_id == "A:1"
    assert candidates["B"][0].chunk_id == "B:1"
    assert all(item.doc_id == "d1" for rows in candidates.values() for item in rows)
