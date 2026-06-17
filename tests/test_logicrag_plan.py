"""LogicRAG 规划 schema / parser / sanitization 的单元测试。"""

import os
import unittest
from unittest.mock import patch

from agent.config import Settings
from agent.reasoning.logicrag import parse_logic_plan, sanitize_logic_plan
from agent.schemas import LogicNode, LogicPlan


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

    def test_settings_from_env_reads_logicrag_fields(self):
        env = {
            "AFAC_LOGICRAG_ENABLED": "true",
            "AFAC_RETRIEVAL_STRATEGY": "logicrag",
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
        self.assertEqual(settings.retrieval_strategy, "logicrag")
        self.assertEqual(settings.logicrag_max_subproblems, 7)
        self.assertEqual(settings.logicrag_max_ranks, 5)
        self.assertEqual(settings.logicrag_rank_top_k, 9)
        self.assertEqual(settings.logicrag_memory_chars, 3200)
        self.assertEqual(settings.logicrag_plan_max_tokens, 640)
        self.assertEqual(settings.logicrag_summary_max_tokens, 288)


if __name__ == "__main__":
    unittest.main()
