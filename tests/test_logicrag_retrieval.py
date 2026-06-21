"""LogicRAG retrieval-first 变体测试。"""

from agent.index.bm25 import BM25SearchIndex
from agent.reasoning import logicrag
from agent.retrieve.variants import get_variant, retrieve_with_variant
from agent.schemas import Chunk, LogicNode, LogicPlan, Question


def _build_question() -> Question:
    return Question(
        qid="q1",
        domain="regulation",
        split="A",
        question="根据《客户尽职调查办法》，2024年银行识别受益所有人的要求是什么？",
        options={
            "A": "识别受益所有人并保存客户身份资料",
            "B": "不需要识别受益所有人",
        },
        answer_format="mcq",
        doc_ids=["doc1"],
    )


def _build_index() -> BM25SearchIndex:
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc1",
            domain="regulation",
            page=1,
            section="",
            clause_id="",
            text="《客户尽职调查办法》要求银行识别受益所有人并保存客户身份资料，2024年继续执行。",
            tables=[],
            numbers=[],
            dates=["2024年"],
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc2",
            domain="regulation",
            page=1,
            section="",
            clause_id="",
            text="证券发行承销管理办法 询价 配售",
            tables=[],
            numbers=[],
            dates=[],
        ),
    ]
    return BM25SearchIndex.build(chunks, tokenizer_mode="mixed")


def test_logicrag_qwen_rrf_is_registered():
    variant = get_variant("logicrag_qwen_rrf")

    assert variant.name == "logicrag_qwen_rrf"
    assert "LogicRAG" in variant.description


def test_build_logicrag_rrf_queries_uses_monkeypatched_planner(monkeypatch):
    question = _build_question()
    planner_calls = []

    def fake_planner(input_question: Question, max_subproblems: int, max_ranks: int):
        planner_calls.append((input_question.qid, max_subproblems, max_ranks))
        return LogicPlan(
            nodes=[
                LogicNode("n1", "定位《客户尽职调查办法》中受益所有人的要求", []),
                LogicNode("n2", "确认是否需要保存客户身份资料", ["n1"]),
                LogicNode("n3", "定位《客户尽职调查办法》中受益所有人的要求", []),
            ],
            rationale="先定位条款，再确认保存要求。",
        )

    monkeypatch.setattr(logicrag, "DEFAULT_LOGIC_PLANNER", fake_planner)

    queries = logicrag.build_logicrag_rrf_queries(question, max_subproblems=4, max_ranks=3)

    assert planner_calls == [("q1", 4, 3)]
    assert queries[0].startswith(question.question)
    assert any("受益所有人" in query for query in queries)
    assert any("保存客户身份资料" in query for query in queries)
    assert len(queries) == len(set(queries))


def test_logicrag_qwen_rrf_retrieves_with_fake_planner(monkeypatch):
    question = _build_question()
    index = _build_index()

    monkeypatch.setattr(
        logicrag,
        "DEFAULT_LOGIC_PLANNER",
        lambda input_question, max_subproblems, max_ranks: LogicPlan(
            nodes=[
                LogicNode("n1", "识别受益所有人", []),
                LogicNode("n2", "保存客户身份资料", ["n1"]),
            ],
            rationale="先识别，再确认保存义务。",
        ),
    )

    results = retrieve_with_variant(index, question, "logicrag_qwen_rrf", top_k=2)

    assert results
    assert results[0].doc_id == "doc1"
    assert results[0].source == "rrf:logicrag_qwen_rrf"


def test_retrieve_rankwise_evidence_propagates_non_empty_memory_anchor():
    question = _build_question()
    plan = LogicPlan(
        nodes=[
            LogicNode("n1", "识别受益所有人", []),
            LogicNode("n2", "保存客户身份资料", ["n1"]),
        ],
        rationale="先识别，再确认保存义务。",
    )
    retriever = type("RetrieverStub", (), {})()
    retriever.index = _build_index()
    retriever._candidate_doc_filter = lambda input_question, restrict_to_doc_ids=True: set(input_question.doc_ids)

    rank_runs, combined = logicrag.retrieve_rankwise_evidence(
        retriever,
        question,
        plan,
        per_query_top_k=2,
        fused_top_k=2,
    )

    assert len(rank_runs) == 2
    assert combined
    assert rank_runs[0]["seed_results"] == []
    assert rank_runs[1]["seed_results"] == []
    assert len(rank_runs[0]["queries"]) == 1
    assert len(rank_runs[1]["queries"]) == 2
    assert rank_runs[0]["packs"]
    assert sum("受益所有人" in query for query in rank_runs[1]["queries"]) >= 2
