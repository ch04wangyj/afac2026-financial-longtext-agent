from agent.retrieve.targets import build_retrieval_target, merge_retrieval_targets, question_with_options
from agent.schemas import Question



def _question() -> Question:
    return Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="比较比亚迪2024年营业收入是否高于2023年。",
        options={"A": "2024年营业收入高于2023年", "B": "2024年营业收入低于2023年"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )



def test_build_retrieval_target_prefers_question_with_options_and_structured_terms():
    question = _question()

    target = build_retrieval_target(question, "定位营业收入并比较2024年与2023年")

    assert target.doc_scope == ["doc1"]
    assert target.query_variants[0] == question_with_options(question)
    assert any("营业收入" in term for term in target.must_terms)
    assert any("2024" in item for item in [*target.numbers, *target.dates, *target.must_terms])
    assert target.evidence_intent in {"comparison", "number"}



def test_merge_retrieval_targets_combines_queries_without_duplicates():
    question = _question()
    target_a = build_retrieval_target(question, "定位营业收入")
    target_b = build_retrieval_target(question, "比较2024年和2023年")

    merged = merge_retrieval_targets(question, [target_a, target_b], node_id="rank_0", rank=0)

    assert merged.node_id == "rank_0"
    assert merged.rank == 0
    assert merged.query_variants
    assert len(merged.query_variants) == len(set(merged.query_variants))
    assert any("营业收入" in query for query in merged.query_variants)
