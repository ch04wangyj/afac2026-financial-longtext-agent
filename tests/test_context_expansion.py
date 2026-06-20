from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.context import build_evidence_packs, select_results_from_packs
from agent.schemas import Chunk, Question, RetrievalResult



def _question() -> Question:
    return Question(
        qid="q1",
        domain="financial_reports",
        split="A",
        question="比较2024年营业收入是否高于2023年。",
        options={"A": "高于", "B": "低于"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )



def test_context_expansion_builds_doc_scoped_pack_for_table_anchor():
    index = BM25SearchIndex.build(
        [
            Chunk("c1", "doc1", "financial_reports", 1, "主要会计数据", "", "2024年营业收入为120亿元。", [], ["120亿元"], ["2024年"]),
            Chunk("c2", "doc1", "financial_reports", 1, "table", "", "表1 主要会计数据 营业收入 120亿元 110亿元", ["营业收入 120亿元 110亿元"], ["120亿元", "110亿元"], [] , {"chunk_type": "table", "title": "年报"}),
            Chunk("c3", "doc1", "financial_reports", 1, "主要会计数据", "", "2023年营业收入为110亿元。", [], ["110亿元"], ["2023年"]),
            Chunk("c4", "doc2", "financial_reports", 1, "其他", "", "其他公司营业收入。", [], [], []),
        ]
    )
    question = _question()
    table_anchor = RetrievalResult(
        chunk_id="c2",
        doc_id="doc1",
        domain="financial_reports",
        score=1.0,
        source="test",
        query="营业收入 2024 2023",
        evidence_text="表1 主要会计数据 营业收入 120亿元 110亿元",
        metadata={"chunk_type": "table", "page": 1, "section": "table", "clause_id": "", "numbers": ["120亿元", "110亿元"], "dates": [], "title": "年报"},
    )

    packs = build_evidence_packs(index, question, [table_anchor], max_packs=3)
    selected = select_results_from_packs(index, question, packs, top_k=6, max_chars=4000)

    assert packs
    assert all(pack.doc_id == "doc1" for pack in packs)
    assert {item.chunk_id for item in selected}.issubset({"c1", "c2", "c3"})
    assert any(item.metadata.get("pack_role") == "anchor" for item in selected)
    assert any(item.metadata.get("pack_role") == "context" for item in selected)
