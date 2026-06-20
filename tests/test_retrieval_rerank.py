from agent.retrieve.rerank import rerank_retrieval_results
from agent.retrieve.targets import build_retrieval_target
from agent.schemas import Question, RetrievalResult



def _question() -> Question:
    return Question(
        qid="q1",
        domain="regulatory",
        split="A",
        question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
        options={"A": "不需要识别受益所有人", "B": "需要识别受益所有人并保存身份资料"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )



def _result(chunk_id: str, text: str, score: float, *, doc_id: str = "doc1", chunk_type: str = "text", section: str = "第一条", clause_id: str = "第一条", domain: str = "regulatory") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain=domain,
        score=score,
        source="test",
        query="客户尽职调查 受益所有人",
        evidence_text=text,
        metadata={
            "numbers": [],
            "dates": [],
            "title": "客户尽职调查办法",
            "section": section,
            "clause_id": clause_id,
            "chunk_type": chunk_type,
        },
    )



def test_rerank_retrieval_results_prefers_must_term_and_structure_matches():
    question = _question()
    target = build_retrieval_target(question, "识别受益所有人并保存身份资料")
    strong = _result("c1", "银行需要识别受益所有人并保存客户身份资料。", 1.0)
    weak = _result("c2", "证券发行承销管理办法 询价 配售", 1.2, doc_id="doc1", chunk_type="figure")

    ranked = rerank_retrieval_results(question, target, [weak, strong])

    assert ranked[0].chunk_id == "c1"
    assert ranked[0].metadata["rerank_score"] >= ranked[1].metadata["rerank_score"]
    assert ranked[0].metadata["rerank_features"]["must_hits"] >= 1



def test_rerank_prefers_specific_clause_consequence_over_generic_principle_page():
    question = Question(
        qid="reg_penalty",
        domain="regulatory",
        split="A",
        question="证券公司因重大违法违规被实施行政处罚的，其分类评价得分是否会被扣减？",
        options={"A": "会", "B": "不会"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    target = build_retrieval_target(question, question.question)
    generic = _result(
        "g1",
        "证券公司分类评价工作应当坚持依法合规、客观公正的原则，设定基准分为100分。",
        1.15,
        section="总则",
        clause_id="第一条",
    )
    specific = _result(
        "s1",
        "证券公司因重大违法违规被实施行政处罚的，应当扣减分类评价得分。",
        1.0,
        section="处罚",
        clause_id="第三十二条",
    )

    ranked = rerank_retrieval_results(question, target, [generic, specific])

    assert ranked[0].chunk_id == "s1"



def test_rerank_prefers_metric_block_over_audit_report_for_financial_comparison():
    question = Question(
        qid="fin_metric",
        domain="financial_reports",
        split="A",
        question="比较两家公司2025年每股现金分红和经营活动现金流净额。",
        options={"A": "每股现金分红更高", "B": "经营活动现金流净额均为正"},
        answer_format="multi",
        doc_ids=["doc1"],
    )
    target = build_retrieval_target(question, question.question)
    audit = _result(
        "a1",
        "我们审计了该公司2025年度财务报表，并认为其公允反映了财务状况。",
        1.2,
        section="审计报告",
        clause_id="",
        domain="financial_reports",
    )
    metric = _result(
        "m1",
        "2025年每10股派现43元，经营活动产生的现金流量净额为120亿元。",
        1.0,
        section="主要财务指标",
        clause_id="",
        domain="financial_reports",
    )

    ranked = rerank_retrieval_results(question, target, [audit, metric])

    assert ranked[0].chunk_id == "m1"
