"""V15 阶段2 版面算法四项深化的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from agent.preprocess.layout_pdf import (
    LayoutParseConfig,
    LayoutTable,
    _apply_chain_continuation,
    _detect_columns_by_clustering,
    _detect_multilevel_header_rows,
    _expand_multilevel_header,
    _fuzzy_margin_signature,
    _is_continuation_relaxed,
    detect_two_column,
    format_table_header,
)


class TestChainContinuation:
    """V15 B1: 跨页表格链式继承。"""

    def test_adjacent_page_continuation_still_works(self):
        """相邻页续表仍能识别。"""
        config = LayoutParseConfig()
        prev = LayoutTable(
            page=10,
            bbox=(50, 100, 500, 700),
            rows=[["2024年", "2023年"], ["100万元", "80万元"], ["200万元", "150万元"]],
            source="pymupdf_lines",
            header=["2024年", "2023年"],
        )
        curr = LayoutTable(
            page=11,
            bbox=(50, 80, 500, 400),
            rows=[["300万元", "250万元"], ["400万元", "350万元"]],
            source="pymupdf_lines",
        )
        _apply_chain_continuation(curr, [prev], config)
        assert curr.continuation is True
        assert curr.header == ["2024年", "2023年"]

    def test_gap_page_continuation_is_detected(self):
        """隔一页的续表能识别（中间是插图页）。"""
        config = LayoutParseConfig(max_continuation_gap=2)
        prev = LayoutTable(
            page=10,
            bbox=(50, 600, 500, 800),
            rows=[["2024年", "2023年", "同比"], ["100万元", "80万元", "25%"]],
            source="pymupdf_lines",
            header=["2024年", "2023年", "同比"],
        )
        curr = LayoutTable(
            page=12,
            bbox=(50, 80, 500, 400),
            rows=[["200万元", "150万元", "33%"], ["300万元", "250万元", "20%"]],
            source="pymupdf_lines",
        )
        _apply_chain_continuation(curr, [prev], config)
        assert curr.continuation is True

    def test_header_variation_tolerated(self):
        """表头列顺序变化仍能识别续表。"""
        config = LayoutParseConfig()
        prev = LayoutTable(
            page=10,
            bbox=(50, 600, 500, 800),
            rows=[["营业收入", "净利润", "总资产"], ["100万元", "50万元", "200万元"]],
            source="pymupdf_lines",
            header=["营业收入", "净利润", "总资产"],
        )
        curr = LayoutTable(
            page=11,
            bbox=(50, 80, 500, 400),
            rows=[["净利润", "总资产", "营业收入"], ["60万元", "210万元", "110万元"]],
            source="pymupdf_lines",
            header=["净利润", "总资产", "营业收入"],
        )
        _apply_chain_continuation(curr, [prev], config)
        assert curr.continuation is True

    def test_unrelated_table_not_continuation(self):
        """不相关的表不被误判为续表。"""
        config = LayoutParseConfig()
        prev = LayoutTable(
            page=10,
            bbox=(50, 100, 500, 200),
            rows=[["指标", "数值"], ["温度", "36.5"]],
            source="pymupdf_lines",
            header=["指标", "数值"],
        )
        curr = LayoutTable(
            page=11,
            bbox=(50, 400, 500, 500),
            rows=[["公司名称", "成立日期"], ["测试公司", "2024年"]],
            source="pymupdf_lines",
        )
        _apply_chain_continuation(curr, [prev], config)
        assert curr.continuation is False


class TestColumnClustering:
    """V15 B2: 双栏 X 坐标聚类检测。"""

    def test_asymmetric_two_column_detected(self):
        """不对称双栏（左短右长）能检测到。"""
        page_width = 500.0
        blocks = [
            ((30, 100, 230, 140), "左栏第一段内容文字" * 3),
            ((30, 150, 230, 190), "左栏第二段内容文字" * 3),
            ((270, 100, 470, 140), "右栏第一段内容文字" * 3),
            ((270, 150, 470, 190), "右栏第二段内容文字" * 3),
            ((270, 200, 470, 240), "右栏第三段内容文字" * 3),
            ((270, 250, 470, 290), "右栏第四段内容文字" * 3),
        ]
        assert _detect_columns_by_clustering(blocks, page_width) is True

    def test_single_column_not_detected(self):
        """单栏不被误判为双栏。"""
        page_width = 500.0
        blocks = [
            ((30, 100, 470, 140), "全文段落内容文字" * 5),
            ((30, 150, 470, 190), "全文段落内容文字" * 5),
            ((30, 200, 470, 240), "全文段落内容文字" * 5),
        ]
        assert _detect_columns_by_clustering(blocks, page_width) is False

    def test_cross_column_block_ignored(self):
        """跨栏标题块不干扰检测。"""
        page_width = 500.0
        blocks = [
            ((30, 50, 470, 80), "跨栏大标题" * 3),
            ((30, 100, 230, 140), "左栏内容文字" * 3),
            ((270, 100, 470, 140), "右栏内容文字" * 3),
            ((30, 150, 230, 190), "左栏内容文字" * 3),
            ((270, 150, 470, 190), "右栏内容文字" * 3),
        ]
        assert _detect_columns_by_clustering(blocks, page_width) is True


class TestFuzzyMargin:
    """V15 B3: 页眉页脚模糊匹配。"""

    def test_page_number_variants_normalized(self):
        """页码变体被归一化为相同签名。"""
        sig1 = _fuzzy_margin_signature("第 5 页")
        sig2 = _fuzzy_margin_signature("第 12 页")
        assert sig1 == sig2

    def test_date_variants_normalized(self):
        """日期变体被归一化为相同签名。"""
        sig1 = _fuzzy_margin_signature("2026年3月15日")
        sig2 = _fuzzy_margin_signature("2026年6月20日")
        assert sig1 == sig2

    def test_pure_number_normalized(self):
        """纯页码被归一化。"""
        sig1 = _fuzzy_margin_signature("5")
        sig2 = _fuzzy_margin_signature("123")
        assert sig1 == sig2

    def test_different_text_different_signature(self):
        """不同文本不会被归一化为相同签名。"""
        sig1 = _fuzzy_margin_signature("招商银行年报")
        sig2 = _fuzzy_margin_signature("中国建筑年报")
        assert sig1 != sig2


class TestMultilevelHeader:
    """V15 B4: 多级表头展开。"""

    def test_three_level_header_expanded(self):
        """三级表头正确展开。"""
        header_rows = [
            ["资产负债表", "资产负债表", "利润表", "利润表"],
            ["流动资产", "非流动资产", "营业收入", "净利润"],
            ["2024年", "2024年", "2024年", "2024年"],
        ]
        data_rows = [["100万元", "200万元", "300万元", "50万元"]]
        result = _expand_multilevel_header(header_rows, data_rows)
        assert len(result) == 4
        assert "资产负债表/流动资产/2024年" in result[0]
        assert "利润表/营业收入/2024年" in result[2]

    def test_single_level_header_not_expanded(self):
        """单级表头不触发多级展开。"""
        rows = [
            ["2024年", "2023年", "同比"],
            ["100万元", "80万元", "25%"],
            ["200万元", "150万元", "33%"],
        ]
        header_rows = _detect_multilevel_header_rows(rows)
        assert len(header_rows) == 0

    def test_two_level_header_detected(self):
        """两级表头被正确检测。"""
        rows = [
            ["资产", "资产", "负债", "负债"],
            ["流动", "非流动", "流动", "非流动"],
            ["100万元", "200万元", "50万元", "30万元"],
        ]
        header_rows = _detect_multilevel_header_rows(rows)
        assert len(header_rows) == 2

    def test_rowspan_inheritance(self):
        """合并单元格（rowspan）值正确继承。"""
        header_rows = [
            ["财务数据", "", "经营数据", ""],
            ["营业收入", "净利润", "订单数", "客户数"],
        ]
        data_rows = [["100万元", "50万元", "1000个", "500个"]]
        result = _expand_multilevel_header(header_rows, data_rows)
        assert len(result) == 4
        # 第一列应继承"财务数据"
        assert "财务数据" in result[0]
        # 第二列也应继承"财务数据"（rowspan）
        assert "财务数据" in result[1]

    def test_format_table_header_with_multilevel(self):
        """format_table_header 传入 table 时正确展开多级表头。"""
        table = LayoutTable(
            page=1,
            bbox=(50, 100, 500, 300),
            rows=[
                ["资产", "资产", "负债", "负债"],
                ["流动", "非流动", "流动", "非流动"],
                ["100万元", "200万元", "50万元", "30万元"],
            ],
            source="pymupdf_lines",
            header=["资产", "资产", "负债", "负债"],
        )
        result = format_table_header(table.header, table=table)
        assert "资产/流动" in result
        assert "负债/非流动" in result
