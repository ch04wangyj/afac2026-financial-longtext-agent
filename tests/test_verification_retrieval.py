from agent.retrieve.claims import build_claim_targets
from agent.retrieve.structured_queries import extract_query_entities
from agent.retrieve.verification_queries import build_verification_query_bundles, extract_predicate_terms
from agent.retrieve.verification_rerank import select_verification_evidence
from agent.schemas import Question, RetrievalResult


def _question() -> Question:
    return Question(
        qid="q1",
        domain="financial_contracts",
        split="a",
        question="关于两份募集说明书的发行规模和主体信用评级，哪些说法正确？",
        options={"A": "第二份文档的发行规模为30亿元", "B": "两份文档主体信用评级均为AAA"},
        answer_format="multi",
        doc_ids=["doc1", "doc2"],
    )


def _result(chunk_id: str, doc_id: str, text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="financial_contracts",
        score=score,
        source="test",
        query="发行规模",
        evidence_text=text,
        metadata={"chunk_type": "atomic_text"},
    )


def test_entity_extraction_does_not_slice_long_option_into_arbitrary_fragments():
    entities = extract_query_entities("第一份文档的发行人名称为广东省广晟控股集团有限公司")

    assert "广东省广晟控股集团有限公司" in entities
    assert "第一份文档的发行" not in entities
    assert "人名称为广东省广" not in entities


def test_predicate_truth_query_excludes_false_candidate_number():
    question = _question()
    claim = build_claim_targets(question)[0]
    bundles = build_verification_query_bundles(question, claim)
    predicate = next(bundle for bundle in bundles if bundle.intent == "predicate_truth")

    assert "发行规模" in predicate.query
    assert "30亿元" not in predicate.query
    assert "30 亿元" not in predicate.query


def test_reranker_keeps_actual_value_as_counter_evidence():
    question = _question()
    claim = build_claim_targets(question)[0]
    predicates = extract_predicate_terms(question, claim)
    candidates = [
        _result("noise", "doc1", "公司近三年有息负债为30亿元。", 1.2),
        _result("truth", "doc2", "发行规模：本期债券总规模不超过5亿元（含5亿元）。", 0.8),
        _result("support", "doc1", "发行规模：本期债券不超过30亿元。", 0.7),
    ]
    selected, report = select_verification_evidence(
        claim,
        candidates,
        predicates,
        top_k=3,
        max_chars=1000,
    )

    assert selected[0].chunk_id in {"truth", "support"}
    assert {item.chunk_id for item in selected} >= {"truth", "support"}
    assert report.roles.get("counter", 0) >= 1
