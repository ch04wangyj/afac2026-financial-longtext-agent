from agent.reasoning.calculation_dsl import build_candidate_calculations, evaluate_calculation
from agent.schemas import Question


def _ledger() -> dict:
    return {
        "facts": [
            {"fact_id": "F1", "doc_id": "a", "metric": "净利润", "year": "2025", "unit": "亿元", "normalized_value": "120", "extraction_mode": "financial_row"},
            {"fact_id": "F2", "doc_id": "a", "metric": "净利润", "year": "2024", "unit": "亿元", "normalized_value": "100", "extraction_mode": "financial_row"},
            {"fact_id": "F3", "doc_id": "b", "metric": "净利润", "year": "2025", "unit": "亿元", "normalized_value": "90", "extraction_mode": "financial_row"},
            {"fact_id": "F4", "doc_id": "b", "metric": "净利润", "year": "2025", "unit": "%", "normalized_value": "0.14", "extraction_mode": "financial_row"},
            {"fact_id": "F5", "doc_id": "b", "metric": "净利润", "year": "2025", "unit": "元", "normalized_value": "999", "extraction_mode": "text_regex"},
        ]
    }


def test_evaluates_whitelisted_growth_and_comparison_with_fact_ids():
    growth = evaluate_calculation(_ledger(), "growth_rate", ["F1", "F2"])
    comparison = evaluate_calculation(_ledger(), "compare", ["F1", "F3"])

    assert growth["result"] == "0.2"
    assert "F1" in growth["expression"]
    assert comparison["result"] == "gt"


def test_rejects_unknown_operation_and_fact_id():
    try:
        evaluate_calculation(_ledger(), "python", ["F1", "F2"])
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported operation must fail")

    try:
        evaluate_calculation(_ledger(), "compare", ["F1", "missing"])
    except KeyError:
        pass
    else:
        raise AssertionError("unknown fact id must fail")


def test_builds_cross_document_compare_and_same_document_growth_candidates():
    question = Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="比较两家公司净利润，甲公司同比增速是否更高？",
        options={"A": "甲公司高于乙公司"},
        answer_format="mcq",
        doc_ids=["a", "b"],
    )

    calculations = build_candidate_calculations(question, _ledger())

    assert any(item["operation"] == "compare" and item["operands"] == ["F1", "F3"] for item in calculations)
    assert any(item["operation"] == "growth_rate" and item["operands"] == ["F1", "F2"] for item in calculations)
    assert not any("F4" in item["operands"] for item in calculations)
    assert not any("F5" in item["operands"] for item in calculations)
