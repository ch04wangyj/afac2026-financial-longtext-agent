"""V15 阶段3 索引构建与融合脚本的单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from agent.preprocess.chunkers import extract_numbers
from agent.schemas import Chunk, TokenUsage, AnswerResult, RetrievalResult

# 导入脚本中的函数（脚本文件名以数字开头，需用 importlib）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "build_v15_index", str(ROOT / "scripts" / "22_build_v15_index.py")
)
build_v15_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_v15_index)


class TestVLChunkCreation:
    """测试 Qwen-VL 提取结果转 Chunk。"""

    def test_vl_chunk_has_correct_metadata(self):
        row = {
            "domain": "research",
            "doc_id": "pack2_text06",
            "file_name": "pack2_text06.pdf",
            "page_index": 11,
            "text": "| 指标 | 2024年 | 2023年 |\n|---|---|---|\n| 营收 | 100万元 | 80万元 |",
            "valid": True,
            "invalid_reason": "",
            "usage": {"prompt_tokens": 1359, "completion_tokens": 671, "total_tokens": 2030},
            "metadata": {"vl_model": "qwen-vl-max", "image_dpi": 150},
        }
        chunk = build_v15_index._make_vl_chunk(row, row["text"])
        assert chunk.doc_id == "pack2_text06"
        assert chunk.domain == "research"
        assert chunk.page == 12  # page_index + 1
        assert chunk.metadata["chunk_type"] == "vl_table_row"
        assert chunk.metadata["parser_name"] == "qwen_vl_offline_v15"
        assert chunk.metadata["vl_model"] == "qwen-vl-max"
        assert "100万元" in chunk.text
        assert len(chunk.numbers) >= 2

    def test_vl_chunk_has_extra_index_fields(self):
        row = {
            "domain": "research",
            "doc_id": "test",
            "file_name": "test.pdf",
            "page_index": 0,
            "text": "| 2024年 | 100万元 |",
            "valid": True,
            "usage": {},
            "metadata": {},
        }
        chunk = build_v15_index._make_vl_chunk(row, row["text"])
        assert "extra_index_fields" in chunk.metadata


class TestMergeV15Chunks:
    """测试 V15 去重合并逻辑。"""

    def test_dedup_keeps_base_and_adds_new(self):
        base = [
            Chunk(chunk_id="b1", doc_id="d1", domain="r", page=1, section="", clause_id="", text="base text one"),
        ]
        vl = [
            Chunk(chunk_id="vl1", doc_id="d1", domain="r", page=2, section="", clause_id="", text="vl table text"),
        ]
        supplement = [
            Chunk(chunk_id="s1", doc_id="d1", domain="r", page=3, section="", clause_id="", text="supplement text"),
        ]
        result = build_v15_index.merge_v15_chunks(base, vl, supplement)
        assert len(result) == 3

    def test_dedup_removes_exact_text_duplicate(self):
        base = [
            Chunk(chunk_id="b1", doc_id="d1", domain="r", page=1, section="", clause_id="", text="same text"),
        ]
        vl = [
            Chunk(chunk_id="vl1", doc_id="d1", domain="r", page=2, section="", clause_id="", text="same text"),
        ]
        supplement: list[Chunk] = []
        result = build_v15_index.merge_v15_chunks(base, vl, supplement)
        assert len(result) == 1

    def test_different_doc_same_text_not_deduped(self):
        base = [
            Chunk(chunk_id="b1", doc_id="d1", domain="r", page=1, section="", clause_id="", text="same text"),
        ]
        vl = [
            Chunk(chunk_id="vl1", doc_id="d2", domain="r", page=2, section="", clause_id="", text="same text"),
        ]
        result = build_v15_index.merge_v15_chunks(base, vl, [])
        assert len(result) == 2


class TestV15MergeScript:
    """测试 V15 保守融合脚本行为（通过 selective_merge 复用）。"""

    def test_empty_reviews_falls_back_all_differences(self, tmp_path):
        from agent.evaluation.selective_merge import merge_candidate_with_baseline

        baseline = [
            _make_answer_result("q1", "A", 100),
            _make_answer_result("q2", "B", 200),
        ]
        candidate = [
            _make_answer_result("q1", "C", 80),
            _make_answer_result("q2", "B", 180),
        ]
        reviews = {}
        merged = merge_candidate_with_baseline(baseline, candidate, reviews)
        # q1 答案不同且无复核 → 回退 A
        assert merged[0].answer == "A"
        # q2 答案相同 → 保留
        assert merged[1].answer == "B"
        # Token 全量继承候选
        assert merged[0].token_usage.total_tokens == 80
        assert merged[1].token_usage.total_tokens == 180

    def test_reviewed_answer_overrides(self, tmp_path):
        from agent.evaluation.selective_merge import merge_candidate_with_baseline

        baseline = [_make_answer_result("q1", "A", 100)]
        candidate = [_make_answer_result("q1", "C", 80)]
        reviews = {"q1": {"answer": "AB", "decision": "direct_source_review", "reason": "test"}}
        merged = merge_candidate_with_baseline(baseline, candidate, reviews)
        assert merged[0].answer == "AB"


def _make_answer_result(qid: str, answer: str, tokens: int) -> AnswerResult:
    return AnswerResult(
        qid=qid,
        answer=answer,
        confidence=0.9,
        evidence=[],
        token_usage=TokenUsage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2, total_tokens=tokens),
        raw_response="",
        metadata={},
    )
