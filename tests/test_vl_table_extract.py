"""V15 Qwen-VL 表格提取模块的单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from agent.preprocess.vl_table_extract import (
    BClassPage,
    VLExtractConfig,
    VLTableResult,
    _is_skip_page,
    render_page_to_png_bytes,
    validate_table_output,
)


class TestValidateTableOutput:
    """测试 Qwen-VL 输出的确定性校验。"""

    def test_empty_output_is_invalid(self):
        valid, reason = validate_table_output("")
        assert not valid
        assert reason == "empty_output"

    def test_not_a_table_is_invalid(self):
        valid, reason = validate_table_output("NOT_A_TABLE")
        assert not valid
        assert reason == "not_a_table"

    def test_too_short_is_invalid(self):
        valid, reason = validate_table_output("short")
        assert not valid
        assert reason == "too_short"

    def test_insufficient_numbers_is_invalid(self):
        valid, reason = validate_table_output("这是一个没有任何数值的表格标题行数据汇总报告")
        assert not valid
        assert reason == "insufficient_numbers"

    def test_no_year_no_unit_is_invalid(self):
        valid, reason = validate_table_output("收入 100 支出 50 利润 30 差额 20 盈余 10")
        assert not valid
        assert reason == "no_year_no_unit"

    def test_valid_table_with_year_and_numbers(self):
        text = "| 指标 | 2024年 | 2023年 |\n|---|---|---|\n| 营业收入 | 1,000万元 | 800万元 |"
        valid, reason = validate_table_output(text)
        assert valid
        assert reason == ""

    def test_valid_table_with_unit_only(self):
        text = "| 项目 | 金额 | 占比 |\n|---|---|---|\n| A | 500万元 | 50% |\n| B | 300万元 | 30% |"
        valid, reason = validate_table_output(text)
        assert valid
        assert reason == ""

    def test_too_long_is_invalid(self):
        text = "1234万元 " * 2000
        valid, reason = validate_table_output(text)
        assert not valid
        assert reason == "too_long"


class TestIsSkipPage:
    """测试章节封面/声明页跳过逻辑。"""

    def test_authorization_page_is_skipped(self):
        assert _is_skip_page("【授权委托书】")

    def test_declaration_page_is_skipped(self):
        assert _is_skip_page("主承销商声明\n本公司已对募集说明书进行了核查")

    def test_chapter_page_is_skipped(self):
        assert _is_skip_page("第三节\n管理层讨论与分析")

    def test_normal_table_page_is_not_skipped(self):
        assert not _is_skip_page("营业收入 1,000万元 同比增长 10%")

    def test_empty_page_is_not_skipped(self):
        assert not _is_skip_page("")


class TestRenderPageToPng:
    """测试 PDF 页面渲染为 PNG。"""

    def test_render_returns_non_empty_png_bytes(self, tmp_path):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 50), "Test table 2024", fontsize=12)
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        png_bytes = render_page_to_png_bytes(pdf_path, 0, dpi=72)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 100
        assert png_bytes[:4] == b"\x89PNG"

    def test_render_high_dpi_produces_larger_image(self, tmp_path):
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 50), "Test", fontsize=12)
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        low_dpi = render_page_to_png_bytes(pdf_path, 0, dpi=72)
        high_dpi = render_page_to_png_bytes(pdf_path, 0, dpi=150)
        assert len(high_dpi) > len(low_dpi)


class TestBClassPageDataclass:
    """测试 B 类页数据结构。"""

    def test_b_class_page_fields(self):
        page = BClassPage(
            domain="financial_reports",
            doc_id="annual_test_2024",
            file_name="annual_test_2024.pdf",
            page_index=17,
            text_preview="建设价值银行",
            image_ratio=1.01,
        )
        assert page.domain == "financial_reports"
        assert page.page_index == 17
        assert page.image_ratio == 1.01


class TestVLTableResultDataclass:
    """测试 VL 提取结果数据结构。"""

    def test_default_usage_is_zero(self):
        result = VLTableResult(
            domain="research",
            doc_id="test",
            file_name="test.pdf",
            page_index=5,
            text="| 指标 | 2024年 |\n| 营收 | 100万元 |",
            valid=True,
        )
        assert result.usage.total_tokens == 0
        assert result.invalid_reason == ""

    def test_invalid_result_carries_reason(self):
        result = VLTableResult(
            domain="research",
            doc_id="test",
            file_name="test.pdf",
            page_index=5,
            text="",
            valid=False,
            invalid_reason="not_a_table",
        )
        assert not result.valid
        assert result.invalid_reason == "not_a_table"
