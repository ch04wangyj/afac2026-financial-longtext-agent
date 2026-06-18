"""Domain-specific evidence coverage expectations for A-board diagnostics."""

from __future__ import annotations



def expected_evidence_facets(domain: str, question_text: str) -> list[str]:
    if domain == "financial_reports":
        return ["entity", "year", "metric", "unit"]
    if domain == "insurance":
        return ["product", "insurance_responsibility", "exclusion"]
    if domain == "regulatory":
        return ["law_article", "case_or_penalty"]
    if domain == "financial_contracts":
        return ["issuer", "amount_or_term", "rating_or_intermediary"]
    if domain == "research":
        return ["entity", "claim", "metric_or_ranking"]
    return []
