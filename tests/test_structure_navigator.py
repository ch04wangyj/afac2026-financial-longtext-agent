from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.structure_navigator import StructureNavigator
from agent.schemas import Chunk


def _chunk(chunk_id: str, page: int | None, text: str, *, section: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="insurance-1",
        domain="insurance",
        page=page,
        section=section,
        clause_id="",
        text=text,
        metadata={"title": "测试保险条款", "chunk_type": "atomic_text"},
    )


def test_structure_navigator_routes_to_relevant_page_and_neighbors():
    index = BM25SearchIndex.build(
        [
            _chunk("p1", 1, "第一章保险责任，约定住院医疗费用。"),
            _chunk("p2", 2, "为防止或者减少损失所支付的必要合理施救费用，最高不超过保险金额。"),
            _chunk("p3", 3, "第三章争议处理和法律适用。"),
            _chunk("p8", 8, "附录释义，与施救费用无关。"),
        ]
    )
    navigator = StructureNavigator(index)

    hits = navigator.search("施救费用 最高不超过保险金额", doc_ids=["insurance-1"])
    expanded = navigator.expand_chunks(hits[:1], page_radius=1)

    assert hits[0].page == 2
    assert {chunk.chunk_id for _, chunk in expanded} == {"p1", "p2", "p3"}


def test_structure_navigator_groups_no_page_chunks_by_section():
    index = BM25SearchIndex.build(
        [
            _chunk("a", None, "客户身份资料应当妥善保存。", section="资料保存"),
            _chunk("b", None, "交易记录保存期限不得少于十年。", section="资料保存"),
            _chunk("c", None, "董事会应当审议定期报告。", section="公司治理"),
        ]
    )
    navigator = StructureNavigator(index)

    hits = navigator.search("交易记录 保存期限", doc_ids=["insurance-1"], top_k_per_doc=1)
    expanded = navigator.expand_chunks(hits)

    assert hits[0].section == "资料保存"
    assert {chunk.chunk_id for _, chunk in expanded} == {"a", "b"}
