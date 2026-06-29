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


def test_research_and_short_financial_aliases_expand_to_source_terms():
    question = Question(
        qid="q2",
        domain="research",
        split="a",
        question="判断新成立基金份额和客户资金杠杆是否正确。",
        options={"A": "主动型新发56.3亿份，客户资金杠杆为4.09倍"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    claim = build_claim_targets(question)[0]
    predicates = extract_predicate_terms(question, claim)

    assert "主动型新发" in predicates
    assert "新成立基金" in predicates
    assert "客户资金杠杆" in predicates


def test_revenue_abbreviation_expands_to_canonical_financial_metric():
    question = Question(
        qid="q3",
        domain="financial_reports",
        split="a",
        question="比较两家公司营收规模。",
        options={"A": "甲公司营收高于乙公司"},
        answer_format="mcq",
        doc_ids=["a", "b"],
    )
    claim = build_claim_targets(question)[0]

    assert "营业收入" in extract_predicate_terms(question, claim)


def test_insurance_clause_terms_expand_to_source_wording():
    question = Question(
        qid="q4",
        domain="insurance",
        split="a",
        question="关于施救费用和保单贷款，下列说法正确的是？",
        options={"A": "施救费用最高不超过保险金额", "B": "允许申请保单贷款"},
        answer_format="multi",
        doc_ids=["a", "b"],
    )
    claims = build_claim_targets(question)

    first = extract_predicate_terms(question, claims[0])
    second = extract_predicate_terms(question, claims[1])

    assert "施救费用" in first
    assert "防止或者减少损失" in first
    assert "保单贷款" in second
    assert "现金价值净额" in second


def test_contract_penalty_formula_expands_to_ground_truth_terms():
    question = Question(
        qid="q5",
        domain="financial_contracts",
        split="a",
        question="以下哪项违约赔偿描述正确？",
        options={"A": "违约赔偿计算公式包含150%的惩罚系数"},
        answer_format="mcq",
        doc_ids=["text03"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert "支付违约金" in predicates
    assert "本金和利息" in predicates


def test_aml_investigation_retention_expands_to_conditional_rule_terms():
    question = Question(
        qid="reg_retention",
        domain="regulatory",
        split="a",
        question="判断反洗钱资料保存期限。",
        options={"A": "若调查在最低保存期限届满时仍未结束，记录应保存至调查结束"},
        answer_format="mcq",
        doc_ids=["strict_v3_009"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert "最低保存期限届满" in predicates
    assert "保存至反洗钱调查工作结束" in predicates


def test_insurance_contract_cancellation_expands_to_identity_threshold_terms():
    question = Question(
        qid="reg_insurance_identity",
        domain="regulatory",
        split="a",
        question="判断身份核验金额门槛。",
        options={"A": "客户申请解除保险合同且退还金额为1万元以上时需核实申请人身份"},
        answer_format="mcq",
        doc_ids=["strict_v3_009"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert "申请解除保险合同" in predicates
    assert "核实申请人身份" in predicates


def test_asset_impairment_notice_uses_option_predicate_not_question_topic():
    question = Question(
        qid="fc_notice",
        domain="financial_contracts",
        split="a",
        question="对比两份文档中的违约责任与特殊条款。",
        options={"A": "若触发资产减值补偿条款，甲方应在报告出具之日起10日内通知"},
        answer_format="mcq",
        doc_ids=["text08"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert "资产减值补偿" in predicates
    assert "减值测试报告" in predicates
    assert "违约责任" not in predicates


def test_security_identity_and_issue_terms_are_first_class_predicates():
    question = Question(
        qid="fc_identity",
        domain="financial_contracts",
        split="a",
        question="比较证券信息。",
        options={
            "A": "两份文档股票代码不同",
            "B": "第二份文档证券简称是安克创新",
            "C": "第一份文档初始转股价为19.59元",
        },
        answer_format="multi",
        doc_ids=["text04", "text07"],
    )
    predicates = [
        extract_predicate_terms(question, claim)
        for claim in build_claim_targets(question)
    ]

    assert "股票代码" in predicates[0]
    assert "证券简称" in predicates[1]
    assert "初始转股价" in predicates[2]


def test_conditional_redemption_keeps_exact_relation_ahead_of_generic_redemption():
    question = Question(
        qid="fc_redemption",
        domain="financial_contracts",
        split="a",
        question="比较可转债条款。",
        options={"A": "两份文档均设置有条件赎回条款"},
        answer_format="mcq",
        doc_ids=["text04", "text05"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert predicates[0] == "有条件赎回条款"
    assert "未转股余额" in predicates


def test_insurance_scenario_terms_expand_to_exclusion_and_formula_wording():
    question = Question(
        qid="ins_scenarios",
        domain="insurance",
        split="a",
        question="判断免责与现金价值公式。",
        options={
            "A": "酒后驾驶导致失能失智护理状态",
            "B": "在不具有接种条件的单位接种后发生异常反应",
            "C": "食品超过保质期导致食物中毒",
            "D": "给出了不同保单年度的比例公式",
        },
        answer_format="multi",
        doc_ids=["1", "7", "13"],
    )
    predicates = [
        extract_predicate_terms(question, claim)
        for claim in build_claim_targets(question)
    ]

    assert "责任免除" in predicates[0]
    assert "预防接种单位" in predicates[1]
    assert "超过规定的保质期限" in predicates[2]
    assert "现金价值等于" in predicates[3]


def test_financial_buyback_relation_expands_to_total_amount_and_profit():
    question = Question(
        qid="fin_buyback",
        domain="financial_reports",
        split="a",
        question="判断资金运作情况。",
        options={"A": "现金分红与股份回购总金额超过归母净利润"},
        answer_format="mcq",
        doc_ids=["annual_midea_2025_report"],
    )
    claim = build_claim_targets(question)[0]

    predicates = extract_predicate_terms(question, claim)

    assert "股份回购" in predicates
    assert "回购总金额" in predicates
    assert "归母净利润" in predicates


def test_hiv_exclusion_and_financial_growth_terms_expand_to_source_rows():
    insurance = Question(
        qid="ins_hiv",
        domain="insurance",
        split="a",
        question="判断免责范围。",
        options={"A": "感染艾滋病病毒（非输血、职业、器官移植）导致重大疾病"},
        answer_format="mcq",
        doc_ids=["4"],
    )
    financial = Question(
        qid="fin_growth",
        domain="financial_reports",
        split="a",
        question="比较两年数据。",
        options={
            "A": "2025年营业总收入的增长率高于2024年",
            "B": "研发投入占营业收入的比例明显提升",
        },
        answer_format="multi",
        doc_ids=["annual_midea_2024_report", "annual_midea_2025_report"],
    )

    insurance_terms = extract_predicate_terms(
        insurance,
        build_claim_targets(insurance)[0],
    )
    financial_terms = [
        extract_predicate_terms(financial, claim)
        for claim in build_claim_targets(financial)
    ]

    assert "感染艾滋病病毒" in insurance_terms
    assert "职业关系" in insurance_terms
    assert "本年比上年增减" in financial_terms[0]
    assert "研发费用占营业收入比例" in financial_terms[1]


def test_financial_reranker_penalizes_single_customer_scope_for_company_revenue():
    question = Question(
        qid="fin_scope",
        domain="financial_reports",
        split="a",
        question="根据年报，以下说法正确的是？",
        options={"A": "比亚迪2024年营业收入为7771.02亿元"},
        answer_format="mcq",
        doc_ids=["annual_byd_2024_report"],
    )
    claim = build_claim_targets(question)[0]
    predicates = extract_predicate_terms(question, claim)
    total = _result(
        "total",
        "annual_byd_2024_report",
        "营业收入合计 777,102,455,000.00 元",
        0.6,
    )
    total.metadata.update(
        {
            "chunk_type": "financial_metric_row",
            "financial_row": {
                "metric": "营业收入",
                "header": "2024年",
                "raw_row": "营业收入合计 777,102,455,000.00 元",
                "cells": [{"year": "2024", "raw_value": "777,102,455,000.00"}],
            },
        }
    )
    customer = _result(
        "customer",
        "annual_byd_2024_report",
        "2024年的营业收入98,561,168千元为对某一单个客户的收入。",
        1.2,
    )
    customer.metadata.update(
        {
            "chunk_type": "financial_metric_row",
            "financial_row": {
                "metric": "营业收入",
                "header": "2024年",
                "raw_row": "2024年的营业收入98,561,168千元为对某一单个客户的收入。",
                "cells": [{"year": "2024", "raw_value": "98,561,168"}],
            },
        }
    )

    selected, _ = select_verification_evidence(
        claim,
        [customer, total],
        predicates,
        top_k=1,
        max_chars=1000,
    )

    assert selected[0].chunk_id == "total"


def test_financial_reranker_prefers_consolidated_over_parent_cashflow_table():
    question = Question(
        qid="fin_cashflow_scope",
        domain="financial_reports",
        split="a",
        question="比较两家公司经营活动现金流。",
        options={"A": "比亚迪经营活动现金流高于另一家公司"},
        answer_format="mcq",
        doc_ids=["annual_byd_2024_report"],
    )
    claim = build_claim_targets(question)[0]
    predicates = extract_predicate_terms(question, claim)
    consolidated = _result(
        "consolidated",
        "annual_byd_2024_report",
        "表名: 合并现金流量表\n数据行: 经营活动产生的现金流量净额 | 133,453,873",
        0.8,
    )
    consolidated.domain = "financial_reports"
    consolidated.metadata["chunk_type"] = "layout_table_row"
    parent = _result(
        "parent",
        "annual_byd_2024_report",
        "表名: 公司现金流量表\n数据行: 经营活动产生的现金流量净额 | 6,155,458",
        1.2,
    )
    parent.domain = "financial_reports"
    parent.metadata["chunk_type"] = "layout_table_row"

    selected, _ = select_verification_evidence(
        claim,
        [parent, consolidated],
        predicates,
        top_k=1,
        max_chars=1000,
    )

    assert selected[0].chunk_id == "consolidated"
