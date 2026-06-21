from agent.reasoning.multi_logicrag import build_gap_aware_retry_query
from agent.schemas import Question, RetrievalResult


def _question() -> Question:
    return Question(
        qid="q_retry_gap",
        domain="financial_reports",
        split="A",
        question="比较比亚迪2024年营业收入是否高于2023年。",
        options={"A": "2024年营业收入高于2023年", "B": "2024年营业收入低于2023年"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )


def test_build_gap_aware_retry_query_keeps_missing_second_endpoint_signal():
    question = _question()
    evidence = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="doc1",
            domain="financial_reports",
            score=1.0,
            source="test",
            query=question.question,
            evidence_text="比亚迪2024年营业收入为100亿元。",
            metadata={},
        )
    ]

    query = build_gap_aware_retry_query(question, "A", "2024年营业收入高于2023年", evidence)

    assert "2023" in query
    assert "营业收入" in query


def test_refinement_should_trigger_on_metric_query_wrong_block():
    from agent.reasoning.retrieval_refiner import should_trigger_retrieval_refinement

    trigger, reason = should_trigger_retrieval_refinement(
        domain="financial_reports",
        sufficiency={
            "failure_tags": ["generic_context_only", "missing_metric_value_pair"],
            "sufficient": False,
        },
    )

    assert trigger is True
    assert reason == "missing_metric_value_pair"


def test_refinement_should_not_trigger_when_sufficiency_is_good():
    from agent.reasoning.retrieval_refiner import should_trigger_retrieval_refinement

    trigger, reason = should_trigger_retrieval_refinement(
        domain="financial_reports",
        sufficiency={
            "failure_tags": [],
            "sufficient": True,
        },
    )

    assert trigger is False
    assert reason == ""


def test_parse_retrieval_refinement_result_extracts_queries_and_goal():
    from agent.reasoning.retrieval_refiner import parse_retrieval_refinement_result

    raw = '''{
      "goal": "改为寻找年报中的具体指标值块",
      "search_intent": "find_metric_value_block",
      "keep_terms": ["比亚迪", "美的集团", "2025"],
      "avoid_terms": ["审计报告", "公司简介"],
      "refined_queries": [
        "比亚迪 2025 归属于上市公司股东的净利润 同比",
        "美的集团 2025 归属于上市公司股东的净利润 同比"
      ]
    }'''

    result = parse_retrieval_refinement_result(raw)

    assert result.goal == "改为寻找年报中的具体指标值块"
    assert result.search_intent == "find_metric_value_block"
    assert len(result.refined_queries) == 2
    assert "归属于上市公司股东的净利润" in result.refined_queries[0]


def test_parse_logicrag_query_bundles_dedupes_and_limits_queries():
    from agent.reasoning.retrieval_refiner import parse_logicrag_query_bundles

    raw = '''{
      "query_bundles": [
        {"query": "比亚迪 2024 营业收入", "intent": "find_metric_value", "evidence_type": "metric_value", "must_terms": ["比亚迪", "营业收入"]},
        {"query": "比亚迪 2024 营业收入", "intent": "duplicate"},
        {"query": "比亚迪 2024 经营活动现金流净额", "intent": "find_metric_value", "evidence_type": "metric_value"}
      ]
    }'''

    bundles = parse_logicrag_query_bundles(raw, max_bundles=4)

    assert [bundle.query for bundle in bundles] == ["比亚迪 2024 营业收入", "比亚迪 2024 经营活动现金流净额"]
    assert bundles[0].intent == "find_metric_value"
    assert bundles[0].must_terms == ["比亚迪", "营业收入"]


def test_parse_logicrag_sufficiency_judgement_normalizes_zero_one():
    from agent.reasoning.retrieval_refiner import parse_logicrag_sufficiency_judgement

    raw = '''{"sufficient": 0, "failure_tags": ["generic_context_only"], "reason": "只有泛页", "missing_evidence": "缺少数值块", "next_search_goal": "查找营业收入具体数值"}'''

    result = parse_logicrag_sufficiency_judgement(raw)

    assert result.sufficient is False
    assert result.failure_tags == ["generic_context_only"]
    assert result.next_search_goal == "查找营业收入具体数值"
