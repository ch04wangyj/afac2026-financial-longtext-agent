from agent.reasoning.evidence_contract import build_evidence_contracts, format_evidence_contracts
from agent.schemas import Question, RetrievalResult


def _question(option: str = "两份文档的营业收入均超过100亿元") -> Question:
    return Question(
        qid="fin_contract",
        domain="financial_reports",
        split="a",
        question="根据两份年报，以下哪些说法正确？",
        options={"A": option},
        answer_format="multi",
        doc_ids=["doc_a", "doc_b"],
    )


def _evidence(doc_id: str, text: str, role: str = "ground_truth") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{doc_id}:{len(text)}",
        doc_id=doc_id,
        domain="financial_reports",
        score=1.0,
        source="test",
        query="营业收入",
        evidence_text=text,
        metadata={"option_key": "A", "verification_role": role},
    )


def test_contract_rejects_universal_claim_with_missing_document_endpoint():
    contracts = build_evidence_contracts(
        _question(),
        [_evidence("doc_a", "营业收入合计为120亿元。")],
    )
    contract = contracts["A"]

    assert contract.missing_doc_ids == ["doc_b"]
    assert contract.missing_numeric_doc_ids == ["doc_b"]
    assert "universal_scope" in contract.risk_tags
    assert contract.selection_ready is False
    assert "文档=doc_b" in format_evidence_contracts(contracts)


def test_contract_accepts_complete_numeric_endpoints_for_both_documents():
    contracts = build_evidence_contracts(
        _question(),
        [
            _evidence("doc_a", "营业收入合计为120亿元。"),
            _evidence("doc_b", "营业收入合计为130亿元。"),
        ],
    )
    contract = contracts["A"]

    assert contract.coverage_score == 1.0
    assert contract.missing_doc_ids == []
    assert contract.missing_predicate_doc_ids == []
    assert contract.missing_numeric_doc_ids == []
    assert contract.selection_ready is True


def test_contract_flags_absence_and_financial_scope_ambiguity():
    question = _question("年报未披露营业收入")
    contracts = build_evidence_contracts(
        question,
        [_evidence("doc_a", "某一单个客户的营业收入为98亿元。")],
    )

    assert "absence_claim" in contracts["A"].risk_tags
    assert "financial_scope_ambiguity" in contracts["A"].risk_tags
    assert contracts["A"].needs_review is True


def test_contract_flags_insurance_coverage_and_deductible_absence_claims():
    for option in ("重疾险不涵盖院外药品费用", "重疾险无免赔额"):
        contracts = build_evidence_contracts(
            _question(option),
            [_evidence("doc_a", "重大疾病保险金按基本保险金额给付。")],
        )

        assert "absence_claim" in contracts["A"].risk_tags
        assert contracts["A"].needs_review is True


def test_contract_does_not_treat_attributable_profit_wording_as_parent_scope():
    question = _question("两家公司归属于上市公司股东的净利润均增长")
    contracts = build_evidence_contracts(
        question,
        [
            _evidence("doc_a", "母公司拥有人应占溢利同比增长34.00%。"),
            _evidence("doc_b", "归属于上市公司股东的净利润同比增长14.29%。"),
        ],
    )

    assert "financial_scope_ambiguity" not in contracts["A"].risk_tags


def test_role_conflict_requires_review_but_does_not_block_complete_evidence():
    contracts = build_evidence_contracts(
        _question("两份文档的营业收入均超过100亿元"),
        [
            _evidence("doc_a", "营业收入合计为120亿元。", "support"),
            _evidence("doc_b", "营业收入合计为130亿元。", "counter"),
        ],
    )

    assert "evidence_conflict" in contracts["A"].risk_tags
    assert contracts["A"].selection_ready is True
    assert contracts["A"].needs_review is True
