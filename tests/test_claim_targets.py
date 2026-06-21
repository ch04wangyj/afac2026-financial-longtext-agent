from agent.schemas import Question


def test_build_claim_targets_creates_one_claim_per_option_for_multi():
    from agent.retrieve.claims import build_claim_targets

    question = Question(
        qid="q_multi",
        domain="financial_reports",
        split="A",
        question="根据年报，以下说法哪些正确？",
        options={"A": "营业收入同比增长", "B": "现金分红下降"},
        answer_format="multi",
        doc_ids=["doc1"],
    )

    claims = build_claim_targets(question)

    assert [claim.option_key for claim in claims] == ["A", "B"]
    assert all(claim.question_id == "q_multi" for claim in claims)
    assert all(claim.doc_scope == ["doc1"] for claim in claims)
    assert claims[0].claim_id == "q_multi:A"
    assert "营业收入同比增长" in claims[0].claim_text



def test_build_claim_targets_creates_one_claim_per_option_for_mcq():
    from agent.retrieve.claims import build_claim_targets

    question = Question(
        qid="q_mcq",
        domain="regulatory",
        split="A",
        question="证券公司被行政处罚后，哪项说法正确？",
        options={"A": "不会扣分", "B": "会扣减分类评价得分"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )

    claims = build_claim_targets(question)

    assert [claim.option_key for claim in claims] == ["A", "B"]
    assert claims[1].claim_type in {"clause_consequence", "fact", "comparison", "date_fact", "metric_fact"}
    assert "会扣减分类评价得分" in claims[1].claim_text



def test_claim_target_converts_to_retrieval_target_for_existing_rerank():
    from agent.retrieve.claims import build_claim_targets, claim_to_retrieval_target

    question = Question(
        qid="q_bridge",
        domain="financial_reports",
        split="A",
        question="比较公司2025年营业收入。",
        options={"A": "营业收入同比增长"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    target = claim_to_retrieval_target(claim)

    assert target.node_id == claim.claim_id
    assert target.question == claim.claim_text
    assert target.doc_scope == claim.doc_scope
    assert target.evidence_intent in {"comparison", "number", "fact"}
