from agent.preprocess.domain_indexing import build_extra_index_fields
from agent.schemas import Chunk


def test_build_extra_index_fields_keeps_company_and_metric_for_financial_reports():
    chunk = Chunk(
        chunk_id="c1",
        doc_id="doc1",
        domain="financial_reports",
        page=1,
        section="管理层讨论与分析",
        clause_id="",
        text="报告期内公司营业收入同比增长，研发投入持续增加。",
        tables=[],
        numbers=["2025 年", "17.04%"],
        dates=[],
        metadata={"title": "比亚迪股份有限公司 2025 年年度报告全文"},
    )

    fields = build_extra_index_fields(chunk)

    assert "营业收入" in fields
    assert any("2025" in item for item in fields)
    assert any("比亚迪" in item for item in fields)


def test_build_extra_index_fields_keeps_law_title_for_regulatory():
    chunk = Chunk(
        chunk_id="c2",
        doc_id="doc2",
        domain="regulatory",
        page=1,
        section="总则",
        clause_id="第一条",
        text="金融机构应当识别受益所有人，并保存客户身份资料。",
        tables=[],
        numbers=[],
        dates=["2025 年 7 月 1 日"],
        metadata={"title": "《金融机构客户尽职调查和客户身份资料及交易记录保存管理办法》"},
    )

    fields = build_extra_index_fields(chunk)

    assert any("《金融机构客户尽职调查和客户身份资料及交易记录保存管理办法》" == item for item in fields)
    assert any("受益所有人" in item for item in fields)
