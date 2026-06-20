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



def _result(chunk_id: str, text: str, score: float, *, doc_id: str = "doc1", chunk_type: str = "text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        domain="regulatory",
        score=score,
        source="test",
        query="客户尽职调查 受益所有人",
        evidence_text=text,
        metadata={
            "numbers": [],
            "dates": [],
            "title": "客户尽职调查办法",
            "section": "第一条",
            "clause_id": "第一条",
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
