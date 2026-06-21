from agent.retrieve.claims import build_claim_targets
from agent.schemas import Question



def test_analyze_claim_evidence_sufficiency_preserves_claim_metadata():
    from agent.reasoning.claim_verifier import analyze_claim_evidence_sufficiency

    question = Question(
        qid="q1",
        domain="regulatory",
        split="A",
        question="证券公司因重大违法违规被处罚是否会扣减分类评价得分？",
        options={"A": "会扣减分类评价得分"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    report = analyze_claim_evidence_sufficiency(
        claim,
        ["证券公司分类评价应坚持依法合规、客观公正的原则。"],
    )

    assert report["claim_id"] == "q1:A"
    assert report["option_key"] == "A"
    assert report["claim_type"] == claim.claim_type
    assert "failure_tags" in report
    assert report["sufficient"] is False



def test_build_claim_refinement_maps_clause_gap_to_clause_query():
    from agent.reasoning.claim_verifier import build_claim_refinement

    question = Question(
        qid="q2",
        domain="regulatory",
        split="A",
        question="证券公司因重大违法违规被处罚是否会扣减分类评价得分？",
        options={"A": "会扣减分类评价得分"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]

    refinement = build_claim_refinement(claim, {"failure_tags": ["missing_clause_consequence"]})

    assert refinement.action == "find_clause_consequence"
    assert any("扣减" in query or "处罚" in query for query in refinement.queries)



def test_assemble_multi_claim_answer_keeps_all_supported_options():
    from agent.reasoning.claim_verifier import assemble_claim_answer

    verdicts = {
        "A": {"relation": "support", "confidence": 0.9},
        "B": {"relation": "refute", "confidence": 0.8},
        "C": {"relation": "support", "confidence": 0.7},
        "D": {"relation": "insufficient", "confidence": 0.2},
    }

    assert assemble_claim_answer(verdicts, answer_format="multi") == "AC"



def test_assemble_single_claim_answer_chooses_strongest_supported_option():
    from agent.reasoning.claim_verifier import assemble_claim_answer

    verdicts = {
        "A": {"relation": "support", "confidence": 0.6},
        "B": {"relation": "support", "confidence": 0.9},
        "C": {"relation": "refute", "confidence": 0.95},
    }

    assert assemble_claim_answer(verdicts, answer_format="mcq") == "B"



def test_should_refine_uncertain_claim_when_insufficient_or_low_confidence():
    from agent.reasoning.claim_verifier import should_refine_claim

    assert should_refine_claim({"sufficient": False}, {"relation": "insufficient", "confidence": 0.2}, threshold=0.7)
    assert should_refine_claim({"sufficient": True}, {"relation": "support", "confidence": 0.4}, threshold=0.7)
    assert not should_refine_claim({"sufficient": True}, {"relation": "support", "confidence": 0.9}, threshold=0.7)
