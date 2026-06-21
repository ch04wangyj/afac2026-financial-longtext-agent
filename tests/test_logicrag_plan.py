"""LogicRAG 规划 schema / parser / sanitization 的单元测试。"""

import os
import unittest
from unittest.mock import patch

from agent.config import Settings
from agent.reasoning.logicrag import append_unresolved_subproblem, parse_logic_plan, sanitize_logic_plan
from agent.reasoning.prompts import build_logicrag_plan_messages
from agent.schemas import LogicNode, LogicPlan, Question


class LogicPlanSchemaTest(unittest.TestCase):
    def test_plan_keeps_topological_ranks(self):
        plan = LogicPlan(
            nodes=[
                LogicNode(node_id="n1", text="识别文档", depends_on=[]),
                LogicNode(node_id="n2", text="比较数值", depends_on=["n1"]),
                LogicNode(node_id="n3", text="汇总结论", depends_on=["n1", "n2"]),
            ]
        )

        self.assertEqual(plan.topological_levels(), [["n1"], ["n2"], ["n3"]])
        self.assertEqual([node.rank for node in plan.nodes], [0, 1, 2])

    def test_parse_logic_plan_from_json(self):
        raw = '{"subproblems":[{"id":"n1","text":"定位财报指标","depends_on":[]},{"id":"n2","text":"比较同比变化","depends_on":["n1"]}],"rationale":"先定位后比较"}'

        plan = parse_logic_plan(raw)

        self.assertEqual([node.node_id for node in plan.nodes], ["n1", "n2"])
        self.assertEqual(plan.rationale, "先定位后比较")
        self.assertEqual(plan.topological_levels(), [["n1"], ["n2"]])

    def test_sanitize_logic_plan_breaks_cycle_dedupes_and_trims(self):
        raw = LogicPlan(
            nodes=[
                LogicNode("n1", "比较 2024 与 2025 营业收入", ["n2"]),
                LogicNode("n2", "定位营业收入", ["n1", "missing"]),
                LogicNode("n3", "定位营业收入", []),
                LogicNode("n4", "   ", []),
                LogicNode("n5", "汇总判断", ["n1", "n2", "n3"]),
            ]
        )

        clean = sanitize_logic_plan(raw, max_subproblems=3, max_ranks=2)

        self.assertEqual([node.node_id for node in clean.nodes], ["n1", "n2"])
        self.assertEqual(clean.topological_levels(), [["n2"], ["n1"]])
        self.assertEqual(clean.nodes[0].depends_on, ["n2"])
        self.assertEqual(clean.nodes[1].depends_on, [])
        self.assertEqual(
            clean.metadata["planner_contract"]["paper_faithful_core"],
            [
                "llm_subproblem_decomposition",
                "logical_dependency_dag",
                "topological_rank_execution",
                "same_rank_merge_ready",
            ],
        )
        self.assertEqual(
            clean.metadata["planner_contract"]["applied_extensions"],
            ["drop_missing_dependencies", "break_cycles", "drop_duplicate_subproblems", "drop_empty_subproblems", "trim_excess_ranks"],
        )

    def test_append_unresolved_subproblem_appends_after_current_rank_sequence(self):
        plan = LogicPlan(
            nodes=[
                LogicNode(node_id="n1", text="定位受益所有人识别义务", depends_on=[]),
                LogicNode(node_id="n2", text="确认是否要求保存身份资料", depends_on=["n1"]),
            ]
        )

        augmented = append_unresolved_subproblem(
            plan,
            text="确认是否存在例外情形",
            depends_on=["n2"],
            append_after_rank=1,
            reason="retrieval_insufficient",
        )

        self.assertEqual(augmented.topological_levels(), [["n1"], ["n2"], ["n3"]])
        self.assertEqual(augmented.node_map()["n3"].depends_on, ["n2"])
        self.assertEqual(augmented.metadata["dynamic_augmentations"][0]["trigger"], "retrieval_insufficient")
        self.assertEqual(augmented.metadata["dynamic_augmentations"][0]["append_after_rank"], 1)

    def test_logicrag_plan_prompt_spells_out_dependency_contract(self):
        question = Question(
            qid="q_logicrag_prompt",
            domain="regulatory",
            split="A",
            question="根据客户尽职调查要求，银行是否必须识别受益所有人并保存身份资料？",
            options={"A": "不需要", "B": "需要"},
            answer_format="mcq",
            doc_ids=["doc1"],
        )

        messages = build_logicrag_plan_messages(question, max_subproblems=4, max_ranks=3)
        text = messages[-1]["content"]

        self.assertIn("depends_on 表示必须先解决的逻辑前置子问题", text)
        self.assertIn("同一依赖层级的子问题后续会合并为一次检索", text)
        self.assertIn("优先把子问题写成可直接检索的目标事实、目标数值、目标日期、目标条件或目标条款", text)
        self.assertIn("不要只把原题换一种说法", text)

    def test_settings_from_env_reads_logicrag_fields(self):
        env = {
            "AFAC_LOGICRAG_ENABLED": "true",
            "AFAC_RETRIEVAL_STRATEGY": "logicrag_agent",
            "AFAC_LOGICRAG_MAX_SUBPROBLEMS": "7",
            "AFAC_LOGICRAG_MAX_RANKS": "5",
            "AFAC_LOGICRAG_RANK_TOP_K": "9",
            "AFAC_LOGICRAG_MEMORY_CHARS": "3200",
            "AFAC_LOGICRAG_PLAN_MAX_TOKENS": "640",
            "AFAC_LOGICRAG_SUMMARY_MAX_TOKENS": "288",
        }
        with patch.dict(os.environ, env, clear=False):
            settings = Settings.from_env()

        self.assertTrue(settings.logicrag_enabled)
        self.assertEqual(settings.retrieval_strategy, "logicrag_agent")
        self.assertEqual(settings.logicrag_max_subproblems, 7)
        self.assertEqual(settings.logicrag_max_ranks, 5)
        self.assertEqual(settings.logicrag_rank_top_k, 9)
        self.assertEqual(settings.logicrag_memory_chars, 3200)
        self.assertEqual(settings.logicrag_plan_max_tokens, 640)
        self.assertEqual(settings.logicrag_summary_max_tokens, 288)

    def test_settings_from_env_uses_doc_first_bm25f_expansion_as_default_retrieval_strategy(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()

        self.assertEqual(settings.retrieval_strategy, "doc_first_bm25f_expansion")


if __name__ == "__main__":
    unittest.main()
