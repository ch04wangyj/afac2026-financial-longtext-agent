from agent.index.bm25 import BM25SearchIndex
from agent.preprocess.hierarchical_chunking import (
    HierarchicalChunkConfig,
    build_hierarchical_corpus,
    infer_strict_clause,
    split_atomic_text,
)


def _row(text: str, *, tables=None, chunk_id="p1") -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "domain": "regulatory",
        "page": 1,
        "section": "第一章 总则",
        "clause_id": "82.91",
        "text": text,
        "tables": tables or [],
        "numbers": [],
        "dates": [],
        "metadata": {"title": "监管办法", "chunk_type": "text"},
    }


def test_strict_clause_does_not_treat_decimal_as_clause_id():
    assert infer_strict_clause("82.91% 为资产负债率") == ""
    assert infer_strict_clause("1,234.56 万元") == ""
    assert infer_strict_clause("1.01 亿元") == ""
    assert infer_strict_clause("1.2 适用范围") == "1.2"
    assert infer_strict_clause("第二十条 机构应当保存资料。") == "第二十条"


def test_atomic_chunks_do_not_merge_across_clause_boundaries():
    text = "第二十条 机构应当保存客户资料。\n保存期限不得少于五年。\n第二十一条 机构应当及时报告。"
    chunks = split_atomic_text(
        text,
        config=HierarchicalChunkConfig(target_chars=80, max_chars=120, min_chars=10),
    )

    assert len(chunks) == 2
    assert chunks[0][2] == "第二十条"
    assert chunks[1][2] == "第二十一条"
    assert all(len(body) <= 120 for body, _, _ in chunks)


def test_short_legal_list_item_is_kept_as_evidence():
    parents, children = build_hierarchical_corpus(
        [_row("第四十六条 股东会依法行使下列职权：\n（二）审议批准董事会的报告；")]
    )

    assert parents
    assert any("审议批准董事会的报告" in chunk.text for chunk in children)


def test_table_rows_repeat_header_and_link_to_parent():
    rows = [
        _row(
            "主要财务数据",
            tables=["指标 | 2025年 | 2024年\n营业收入 | 120亿元 | 100亿元\n净利润 | 12亿元 | 10亿元"],
        )
    ]
    parents, children = build_hierarchical_corpus(rows)
    table_rows = [chunk for chunk in children if chunk.metadata.get("chunk_type") == "table_row"]

    assert len(parents) == 1
    assert len(table_rows) == 2
    assert all("表头: 指标 | 2025年 | 2024年" in chunk.text for chunk in table_rows)
    assert all(chunk.metadata["parent_chunk_id"] == "p1" for chunk in table_rows)


def test_index_keeps_parent_out_of_scoring_but_can_restore_it():
    parents, children = build_hierarchical_corpus(
        [_row("第二十条 机构应当保存客户资料，保存期限不得少于五年。")],
        HierarchicalChunkConfig(min_chars=10),
    )
    index = BM25SearchIndex.build(children, parent_chunks=parents)
    results = index.search(
        "保存期限 五年",
        top_k=3,
        filter_chunk_types={"atomic_text"},
        scoring_mode="bm25f_lite",
    )

    assert results
    assert all(item.metadata["chunk_type"] == "atomic_text" for item in results)
    assert index.get_parent_chunk(results[0].chunk_id).chunk_id == "p1"
