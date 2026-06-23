import importlib.util
from pathlib import Path

from agent.schemas import Chunk


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "19_build_layout_index.py"
    spec = importlib.util.spec_from_file_location("build_layout_index", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _chunk(chunk_id: str, text: str, chunk_type: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        domain="financial_reports",
        page=1,
        section="",
        clause_id="",
        text=text,
        metadata={"chunk_type": chunk_type},
    )


def test_merge_layout_supplement_keeps_base_order_and_deduplicates_exact_text():
    module = _load_script_module()
    base = [_chunk("base", "营业收入 100 亿元", "atomic_text")]
    supplement = [
        _chunk("duplicate", "营业收入\n100 亿元", "layout_text"),
        _chunk("table", "表头: 2025年 | 2024年\n数据行: 净利润 | 20 | 10", "layout_table_row"),
    ]

    merged = module.merge_supplement_chunks(base, supplement)

    assert [chunk.chunk_id for chunk in merged] == ["base", "table"]
