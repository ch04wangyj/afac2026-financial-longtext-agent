from agent.domain.coverage_rules import expected_evidence_facets



def test_financial_report_question_requires_metric_year_company_facets():
    facets = expected_evidence_facets(domain="financial_reports", question_text="比较比亚迪2024年和美的2025年营业收入")
    assert "metric" in facets
    assert "year" in facets
    assert "entity" in facets



def test_insurance_question_requires_responsibility_and_exclusion_facets():
    facets = expected_evidence_facets(domain="insurance", question_text="哪些事故会导致赔付")
    assert "insurance_responsibility" in facets
    assert "exclusion" in facets
