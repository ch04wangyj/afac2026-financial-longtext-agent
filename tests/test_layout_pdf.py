from types import SimpleNamespace

from agent.preprocess.layout_pdf import (
    LayoutParseConfig,
    LayoutTable,
    VisualLine,
    _extract_borderless_tables,
    _is_continuation,
    _merge_visual_rows,
    _normalize_grid_rows,
    _spans_to_cells,
    detect_two_column,
    format_table_header,
    format_table_row,
)


def _span(text: str, x0: float, x1: float, size: float = 10.0) -> dict:
    return {"text": text, "bbox": [x0, 10, x1, 20], "size": size, "flags": 0}


def test_spans_to_cells_uses_large_coordinate_gap_as_column_boundary():
    cells = _spans_to_cells(
        [
            _span("营业收入", 10, 55),
            _span("1,050,187", 180, 230),
            _span("1,040,759", 300, 350),
            _span("0.9%", 420, 445),
        ]
    )

    assert cells == ["营业收入", "1,050,187", "1,040,759", "0.9%"]


def test_borderless_table_repeats_year_header_and_rows():
    lines = [
        VisualLine((10, 10, 200, 20), "财务概览", ("财务概览",), 14, True),
        VisualLine((10, 30, 450, 40), "2025年2024年变化", ("2025年", "2024年", "变化")),
        VisualLine((10, 50, 450, 60), "营业收入1,050,1871,040,7590.9%", ("营业收入", "1,050,187", "1,040,759", "0.9%")),
        VisualLine((10, 70, 450, 80), "净利润137,095138,373-0.9%", ("净利润", "137,095", "138,373", "-0.9%")),
    ]

    tables = _extract_borderless_tables(lines, 1, LayoutParseConfig())

    assert len(tables) == 1
    assert tables[0].header == ["2025年", "2024年", "变化"]
    assert tables[0].caption == "财务概览"
    assert "表头: 2025年 | 2024年 | 变化" in format_table_row(tables[0], tables[0].rows[1])


def test_visual_cells_on_same_y_axis_are_merged_into_one_row():
    lines = [
        VisualLine((10, 20, 80, 30), "营业收入", ("营业收入",)),
        VisualLine((180, 20.5, 240, 30.5), "1,050,187", ("1,050,187",)),
        VisualLine((300, 20.2, 360, 30.2), "1,040,759", ("1,040,759",)),
    ]

    merged = _merge_visual_rows(lines)

    assert len(merged) == 1
    assert merged[0].cells == ("营业收入", "1,050,187", "1,040,759")


def test_complementary_sparse_grid_rows_are_joined_cell_by_cell():
    rows = _normalize_grid_rows(
        [
            ["", "407,149,600", "", "372,037,280", "", "9.44%"],
            ["营业收入合计", None, "100.00%", None, "100.00%", None],
        ]
    )

    assert rows == [["营业收入合计", "407,149,600", "100.00%", "372,037,280", "100.00%", "9.44%"]]


def test_multilevel_year_header_is_expanded_to_value_columns():
    header = ["2024年", "2023年", "同比增减", "金额", "占营业收入比重"]

    assert format_table_header(header) == (
        "2024年-金额 | 2024年-占营业收入比重 | 2023年-金额 | "
        "2023年-占营业收入比重 | 同比增减"
    )


def test_cross_page_table_inherits_previous_header():
    previous = LayoutTable(
        page=2,
        bbox=(0, 100, 500, 700),
        rows=[["项目", "2025年", "2024年"], ["营业收入", "10", "9"]],
        source="test",
        header=["项目", "2025年", "2024年"],
    )
    current = LayoutTable(
        page=3,
        bbox=(0, 50, 500, 700),
        rows=[["净利润", "2", "1"]],
        source="test",
    )

    assert _is_continuation(previous, current) is True


def test_adjacent_tables_in_middle_of_pages_are_not_false_continuations():
    previous = LayoutTable(
        page=2,
        bbox=(0, 200, 500, 500),
        rows=[["项目", "金额"], ["收入", "10"]],
        source="test",
        header=["项目", "金额"],
    )
    current = LayoutTable(
        page=3,
        bbox=(0, 220, 500, 520),
        rows=[["员工", "20"]],
        source="test",
    )

    assert _is_continuation(previous, current) is False


def test_two_column_detection_requires_both_sides_and_vertical_overlap():
    blocks = [
        ((20, 100, 250, 240), "左栏" * 80),
        ((20, 250, 250, 400), "左栏正文" * 50),
        ((350, 110, 580, 250), "右栏" * 80),
        ((350, 260, 580, 410), "右栏正文" * 50),
    ]

    assert detect_two_column(blocks, 600) is True
    assert detect_two_column(blocks[:2], 600) is False
