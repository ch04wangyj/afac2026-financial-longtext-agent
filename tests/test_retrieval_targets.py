from agent.retrieve.targets import analyze_evidence_sufficiency, build_retrieval_target, merge_retrieval_targets, question_with_options
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



def test_analyze_evidence_sufficiency_flags_missing_comparison_coverage():
    question = _question()
    target = build_retrieval_target(question, "定位营业收入并比较2024年与2023年")

    report = analyze_evidence_sufficiency(
        target,
        evidence_texts=[
            "比亚迪 2024 年营业收入 100 亿元。",
            "本报告披露公司营业收入持续增长。",
        ],
    )

    assert report["sufficient"] is False
    assert report["comparison_incomplete"] is True
    assert report["missing_numbers"]



def test_sufficiency_v2_flags_missing_second_endpoint_for_comparison():
    question = _question()
    target = build_retrieval_target(question, "比较比亚迪2024年营业收入是否高于2023年")

    report = analyze_evidence_sufficiency(
        target,
        evidence_texts=["比亚迪 2024 年营业收入为 100 亿元。"],
    )

    assert report["sufficient"] is False
    assert "missing_second_endpoint" in report["failure_tags"]



def test_sufficiency_v2_flags_generic_context_only_when_no_adjudicable_fact():
    question = _question()
    target = build_retrieval_target(question, "比较比亚迪2024年营业收入是否高于2023年")

    report = analyze_evidence_sufficiency(
        target,
        evidence_texts=["我们审计了公司2025年度财务报表，并认为其公允反映了财务状况。"],
    )

    assert report["sufficient"] is False
    assert "generic_context_only" in report["failure_tags"]



def test_sufficiency_v2_flags_missing_clause_consequence_for_regulatory_prompt():
    question = Question(
        qid="reg1",
        domain="regulatory",
        split="A",
        question="根据监管规定，证券公司因重大违法违规被实施行政处罚的，其分类评价得分是否会被扣减？",
        options={"A": "会", "B": "不会"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    target = build_retrieval_target(question, question.question)

    report = analyze_evidence_sufficiency(
        target,
        evidence_texts=["证券公司分类评价工作应当坚持依法合规、客观公正的原则，设定基准分为100分。"],
    )

    assert report["sufficient"] is False
    assert "missing_clause_consequence" in report["failure_tags"]
