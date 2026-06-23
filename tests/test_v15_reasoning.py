"""V15 单元测试：PoT 推理 + 自验证 + 自适应路由 + LLM Judge。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from agent.reasoning.adaptive_router import (
    RouterConfig,
    RoutingDecision,
    llm_judge_rerank,
    route_question,
)
from agent.reasoning.pot_reasoner import (
    PoTConfig,
    PoTResult,
    _execute_dsl,
    _to_decimal,
    format_pot_results,
    needs_pot,
)
from agent.reasoning.self_verifier import (
    SelfVerifierConfig,
    VerificationCheck,
    build_query_refine_messages,
    build_verification_messages,
)
from agent.schemas import Question, RetrievalResult, TokenUsage


def _make_question(qid="test_001", fmt="multi", question_text="哪个公司营收更高？", options=None):
    return Question(
        qid=qid,
        domain="financial_reports",
        split="a",
        question=question_text,
        options=options or {"A": "公司A营收100万", "B": "公司B营收200万"},
        answer_format=fmt,
        type="",
        doc_ids=["doc_a", "doc_b"],
    )


class TestNeedsPot:
    """测试 PoT 需求检测。"""

    def test_comparison_question_needs_pot(self):
        q = _make_question(question_text="比亚迪2024年营收是否高于美的集团？")
        assert needs_pot(q) is True

    def test_plain_question_does_not_need_pot(self):
        q = _make_question(question_text="公司名称是什么？", options={"A": "比亚迪", "B": "美的"})
        assert needs_pot(q) is False

    def test_ratio_question_needs_pot(self):
        q = _make_question(question_text="研发投入占营业收入的比例是多少？")
        assert needs_pot(q) is True

    def test_growth_rate_question_needs_pot(self):
        q = _make_question(question_text="2025年净利润同比增长率是多少？")
        assert needs_pot(q) is True


class TestDSLExecution:
    """测试受限 DSL 执行。"""

    def test_compare_greater(self):
        facts = {"F1": {"normalized_value": "2000000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("compare(F1, F2)", facts)
        assert result == "greater"

    def test_compare_less(self):
        facts = {"F1": {"normalized_value": "500000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("compare(F1, F2)", facts)
        assert result == "less"

    def test_compare_equal(self):
        facts = {"F1": {"normalized_value": "1000000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("compare(F1, F2)", facts)
        assert result == "equal"

    def test_difference(self):
        facts = {"F1": {"normalized_value": "2000000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("difference(F1, F2)", facts)
        assert "1000000" in result

    def test_ratio(self):
        facts = {"F1": {"normalized_value": "2000000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("ratio(F1, F2)", facts)
        assert "2" in result

    def test_growth_rate(self):
        facts = {"F1": {"normalized_value": "1200000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("growth_rate(F1, F2)", facts)
        assert "20.00%" in result

    def test_growth_rate_negative(self):
        facts = {"F1": {"normalized_value": "800000"}, "F2": {"normalized_value": "1000000"}}
        result = _execute_dsl("growth_rate(F1, F2)", facts)
        assert "-20.00%" in result

    def test_unknown_op_returns_none(self):
        facts = {"F1": {"normalized_value": "100"}, "F2": {"normalized_value": "200"}}
        result = _execute_dsl("multiply(F1, F2)", facts)
        assert result is None

    def test_missing_fact_returns_none(self):
        facts = {"F1": {"normalized_value": "100"}}
        result = _execute_dsl("compare(F1, F99)", facts)
        assert result is None

    def test_division_by_zero(self):
        facts = {"F1": {"normalized_value": "100"}, "F2": {"normalized_value": "0"}}
        result = _execute_dsl("ratio(F1, F2)", facts)
        assert result == "incomparable"


class TestToDecimal:
    """测试安全 Decimal 转换。"""

    def test_plain_number(self):
        assert _to_decimal("12345") is not None

    def test_comma_separated(self):
        assert _to_decimal("1,234,567") is not None

    def test_chinese_comma(self):
        assert _to_decimal("1，234，567") is not None

    def test_invalid_returns_none(self):
        assert _to_decimal("abc") is None

    def test_empty_returns_none(self):
        assert _to_decimal("") is None


class TestFormatPoTResults:
    """测试 PoT 结果格式化。"""

    def test_empty_executions_returns_empty(self):
        result = PoTResult(program="", executions=[], verified=False, usage=TokenUsage())
        assert format_pot_results(result) == ""

    def test_with_executions(self):
        result = PoTResult(
            program="test",
            executions=[{"option": "A", "dsl": "compare(F1, F2)", "result": "greater", "reason": "比较营收"}],
            verified=True,
            usage=TokenUsage(),
        )
        formatted = format_pot_results(result)
        assert "PoT" in formatted
        assert "选项A" in formatted
        assert "greater" in formatted


class TestRouteQuestion:
    """测试自适应策略路由。"""

    def test_multi_question_routes_to_thinking(self):
        q = _make_question(fmt="multi", question_text="下列哪些选项是正确的？")
        decision = route_question(q)
        assert decision.strategy == "complex_thinking"
        assert decision.enable_thinking is True

    def test_mcq_question_routes_to_simple(self):
        q = _make_question(fmt="mcq", question_text="公司名称是什么？", options={"A": "比亚迪", "B": "美的", "C": "宁德", "D": "移动"})
        decision = route_question(q)
        assert decision.strategy == "simple_cot"
        assert decision.enable_thinking is False

    def test_pot_question_routes_to_pot(self):
        q = _make_question(fmt="multi", question_text="比亚迪2024年营收是否高于美的集团？")
        decision = route_question(q)
        assert decision.strategy == "pot"
        assert decision.enable_thinking is True

    def test_tf_question_routes_to_simple(self):
        q = _make_question(fmt="tf", question_text="公司名称是比亚迪", options={"A": "正确", "B": "错误"})
        decision = route_question(q)
        assert decision.strategy == "simple_cot"


class TestSelfVerifier:
    """测试自验证模块。"""

    def test_high_confidence_skips_verification(self):
        from agent.reasoning.self_verifier import run_self_verification

        # mock LLM
        class MockLLM:
            settings = type("S", (), {"qwen_model": "test"})()
            def chat(self, *a, **kw):
                pass

        q = _make_question()
        result = run_self_verification(
            q, "context", "A", 0.95, MockLLM(), config=SelfVerifierConfig(confidence_threshold=0.8)
        )
        assert result.accepted is True
        assert result.iterations == 0

    def test_verification_messages_have_correct_structure(self):
        q = _make_question()
        messages = build_verification_messages(q, "evidence context", "A", 0.5)
        assert len(messages) == 2
        assert "system" in messages[0]["role"]
        assert "ACCEPT" in messages[1]["content"] or "REJECT" in messages[1]["content"]

    def test_refine_messages_have_strategy(self):
        q = _make_question()
        messages = build_query_refine_messages(q, ["选项C缺少数值"], "synonym_expansion")
        assert len(messages) == 2
        assert "同义词" in messages[0]["content"]


class TestLLMJudgeRerank:
    """测试 LLM-as-a-Judge 重排。"""

    def test_few_candidates_returned_directly(self):
        class MockLLM:
            settings = type("S", (), {"qwen_model": "test"})()
            def chat(self, *a, **kw):
                pass

        candidates = [
            RetrievalResult(chunk_id="c1", doc_id="d1", domain="r", score=1.0, source="s", query="q", evidence_text="text1"),
            RetrievalResult(chunk_id="c2", doc_id="d2", domain="r", score=0.8, source="s", query="q", evidence_text="text2"),
        ]
        q = _make_question()
        result, usage = llm_judge_rerank(q, candidates, MockLLM(), RouterConfig(judge_top_k_output=5))
        assert len(result) == 2
        assert usage.total_tokens == 0  # MockLLM not called
