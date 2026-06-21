from agent.reasoning.prompts import build_financial_metric_extraction_messages
from agent.domain.financial_reports import compare_growth, normalize_numeric_value, ratio_exceeds
from agent.schemas import Question, RetrievalResult



def _question() -> Question:
    return Question(
        qid="fin_calc_q1",
        domain="financial_reports",
        split="A",
        question="根据比亚迪2024年和美的2025年财报披露，下列说法是否正确？",
        options={
            "A": "比亚迪2024年经营活动现金流净额低于营业收入的一半",
            "B": "美的2025年经营活动现金流净额超过营业收入的十分之一",
        },
        answer_format="multi",
        type="财务指标对比分析",
        doc_ids=["annual_byd_2024_report", "annual_midea_2025_report"],
    )



def _evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="byd:1",
            doc_id="annual_byd_2024_report",
            domain="financial_reports",
            score=1.0,
            source="test",
            query="经营活动现金流量净额 营业收入",
            evidence_text="2024 年营业收入(千元) 777,102,000；经营活动产生的现金流量净额(千元) 133,453,873。",
            metadata={"title": "比亚迪2024年年度报告", "page": 32},
        ),
        RetrievalResult(
            chunk_id="midea:1",
            doc_id="annual_midea_2025_report",
            domain="financial_reports",
            score=1.0,
            source="test",
            query="经营活动现金流量净额 营业收入",
            evidence_text="2025 年营业收入(千元) 456,451,731；经营活动产生的现金流量净额(千元) 53,345,930。",
            metadata={"title": "美的2025年年度报告", "page": 53},
        ),
    ]



def test_normalize_thousand_yuan_to_yuan():
    assert normalize_numeric_value("456,451,731", unit="千元") == 456451731000



def test_normalize_billion_yuan_to_yuan():
    assert normalize_numeric_value("1.23", unit="亿元") == 123000000



def test_compare_growth_detects_increase():
    assert compare_growth(current=120, previous=100) == "increase"



def test_ratio_exceeds_threshold():
    assert ratio_exceeds(numerator=55, denominator=500, threshold=0.1) is True



def test_financial_metric_extraction_prompt_demands_units_and_evidence_ids():
    messages = build_financial_metric_extraction_messages(_question(), _evidence())
    text = messages[-1]["content"]
    assert "metric_values" in text
    assert "unit" in text
    assert "evidence_id" in text
    assert "不要计算最终答案" in text
