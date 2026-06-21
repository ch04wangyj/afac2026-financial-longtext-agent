from agent.preprocess.financial_rows import extract_financial_metric_rows, format_financial_metric_row


def test_extracts_table_header_metric_values_units_and_years():
    text = """六、主要会计数据和财务指标
单位：元
2025 年 2024 年 本年比上年增减 2023 年
营业收入(元) 803,964,958,000.00 777,102,455,000.00 3.46% 602,315,354,000.00
归属于上市公司股东的净利润(元) 32,619,022,000.00 40,254,346,000.00 -18.97% 30,040,811,000.00
"""

    rows = extract_financial_metric_rows(text)

    assert [row["metric"] for row in rows] == ["营业收入", "归属于上市公司股东的净利润"]
    assert rows[0]["unit"] == "元"
    assert rows[0]["cells"][0] == {
        "column": "2025 年",
        "year": "2025",
        "raw_value": "803,964,958,000.00",
        "unit": "元",
    }
    assert rows[0]["cells"][2]["column"] == "本年比上年增减"
    assert rows[0]["cells"][2]["year"] == "2025"
    assert rows[0]["cells"][2]["unit"] == "%"


def test_joins_wrapped_metric_row_and_preserves_parenthesized_negative():
    text = """2025 年 2024 年
经营活动产生的现金流量净额(千元)
(57,967,687) 53,345,930
"""

    rows = extract_financial_metric_rows(text)

    assert len(rows) == 1
    assert rows[0]["cells"][0]["raw_value"] == "(57,967,687)"
    assert rows[0]["cells"][0]["year"] == "2025"
    assert rows[0]["cells"][0]["unit"] == "千元"


def test_formats_compact_retrieval_text_without_losing_raw_row():
    row = extract_financial_metric_rows("2025 年研发投入约634亿元，同比上升17%。")[0]
    text = format_financial_metric_row(row, title="比亚迪2025年年度报告")

    assert "财务指标: 研发投入" in text
    assert "634亿元" in text
    assert "17%" in text


def test_matches_financial_metric_names_with_pdf_spacing():
    rows = extract_financial_metric_rows("每10 股派息数(元)(含税) 3.58")

    assert rows[0]["metric"] == "每10股现金分红"
    assert rows[0]["cells"][0]["raw_value"] == "3.58"


def test_narrative_metric_does_not_inherit_unrelated_prior_table_header():
    text = """2024 年 2023 年
其他表格行 10 9
于2024年度，本集团的营业收入在某一时点确认的金额为380,685百万元；在某一时段确认的金额为26,346百万元。
"""

    row = extract_financial_metric_rows(text)[0]

    assert row["header"] == ""
    assert {cell["year"] for cell in row["cells"]} == {"2024"}
    assert row["cells"][0]["unit"] == "百万元"


def test_narrative_single_value_does_not_consume_next_unrelated_line():
    text = "归母净利润439.5亿元，收入和利润均实现双位数增长。海外收入1,959亿元，同比增长10%。"

    row = extract_financial_metric_rows(text, default_year="2025")[0]

    assert row["cells"] == [{"column": "", "year": "2025", "raw_value": "439.5", "unit": "亿元"}]
