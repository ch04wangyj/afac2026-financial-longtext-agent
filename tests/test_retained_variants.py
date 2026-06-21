from __future__ import annotations

import pytest

from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.variants import get_variant, retrieve_with_variant
from agent.schemas import Chunk, LogicNode, LogicPlan, Question
from agent.reasoning import logicrag


@pytest.fixture
def question() -> Question:
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


@pytest.fixture
def index() -> BM25SearchIndex:
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


def test_only_retained_variants_exist():
    assert get_variant("doc_first_bm25f_expansion").name == "doc_first_bm25f_expansion"
    assert get_variant("logicrag_qwen_rrf").name == "logicrag_qwen_rrf"

    for legacy in [
        "question_options",
        "rule_multi_rrf",
        "field_boosted_rrf",
        "logic_lite_rrf",
        "linear_entity_rrf",
        "graph_lite_rrf",
        "crag_lite",
        "oracle_doc_restricted",
        "bm25f_lite_rrf",
        "broad_sparse_structured_rerank",
    ]:
        with pytest.raises(KeyError):
            get_variant(legacy)


def test_doc_first_variant_retrieves_target_doc(index: BM25SearchIndex, question: Question):
    results = retrieve_with_variant(index, question, "doc_first_bm25f_expansion", top_k=2)
    assert results
    assert results[0].doc_id == "doc1"
    assert results[0].source == "doc_first_bm25f_expansion"


def test_logicrag_qwen_rrf_variant_retrieves_target_doc(monkeypatch: pytest.MonkeyPatch, index: BM25SearchIndex, question: Question):
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
