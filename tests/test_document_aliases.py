from agent.data.document_aliases import document_label, option_doc_scope
from agent.schemas import Question


def _insurance_question() -> Question:
    return Question(
        qid="ins-test",
        domain="insurance",
        split="a",
        question="关于保单贷款，哪些说法正确？",
        options={
            "A": "平安智盈金生允许保单贷款",
            "B": "平安富鸿金生在个人养老金制度下不允许保单贷款",
            "C": "比较平安智盈金生与平安富鸿金生",
        },
        answer_format="multi",
        doc_ids=["1", "2", "16"],
    )


def test_option_doc_scope_uses_public_product_aliases():
    question = _insurance_question()

    assert option_doc_scope(question, question.options["A"]) == ["1"]
    assert option_doc_scope(question, question.options["B"]) == ["16"]
    assert option_doc_scope(question, question.options["C"]) == ["1", "16"]


def test_option_doc_scope_falls_back_when_no_alias_matches():
    question = _insurance_question()

    assert option_doc_scope(question, "所有产品均允许贷款") == ["1", "2", "16"]


def test_document_label_replaces_opaque_insurance_id():
    assert document_label("insurance", "9", "9") == "平安特种车险"


def test_option_doc_scope_matches_short_insurer_alias():
    question = _insurance_question()

    assert option_doc_scope(question, "太保赔付0.2万元") == ["1", "2", "16"]

    medical_question = Question(
        qid="ins-medical",
        domain="insurance",
        split="a",
        question="比较两款医疗险。",
        options={"A": "太保赔付0.2万元"},
        answer_format="mcq",
        doc_ids=["5", "6"],
    )
    assert option_doc_scope(medical_question, medical_question.options["A"]) == ["6"]


def test_option_doc_scope_matches_contract_public_question_id():
    question = Question(
        qid="fc-test",
        domain="financial_contracts",
        split="a",
        question="比较两份文档。",
        options={"A": "文档 fc_text_003 的违约赔偿公式"},
        answer_format="mcq",
        doc_ids=["text02", "text03"],
    )

    assert option_doc_scope(question, question.options["A"]) == ["text03"]


def test_option_doc_scope_uses_year_to_disambiguate_financial_reports():
    question = Question(
        qid="fin-test",
        domain="financial_reports",
        split="a",
        question="比较比亚迪两年年报。",
        options={"A": "比亚迪2025年营业收入增长"},
        answer_format="mcq",
        doc_ids=["annual_byd_2024_report", "annual_byd_2025_report"],
    )

    assert option_doc_scope(question, question.options["A"]) == ["annual_byd_2025_report"]
