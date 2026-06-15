"""提交审计产物测试。"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from agent.schemas import AnswerResult, RetrievalResult, TokenUsage


ROOT = Path(__file__).resolve().parents[1]


def _load_submission_module():
    """从脚本路径加载 04_make_submission.py，避免数字文件名影响普通 import。"""
    path = ROOT / "scripts" / "04_make_submission.py"
    spec = importlib.util.spec_from_file_location("make_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SubmissionArtifactsTest(unittest.TestCase):
    """校验 evidence.json 的单条证据字段完整性。"""

    def test_evidence_items_include_qid(self) -> None:
        module = _load_submission_module()
        result = AnswerResult(
            qid="q1",
            answer="A",
            confidence=0.9,
            evidence=[
                RetrievalResult(
                    chunk_id="c1",
                    doc_id="d1",
                    domain="financial_reports",
                    score=1.0,
                    source="test",
                    query="question",
                    evidence_text="关键证据",
                )
            ],
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        )

        rows = module.evidence_items_with_qid(result)

        self.assertEqual(rows[0]["qid"], "q1")
        self.assertEqual(rows[0]["doc_id"], "d1")
        self.assertEqual(rows[0]["chunk_id"], "c1")
        self.assertEqual(rows[0]["evidence_text"], "关键证据")


if __name__ == "__main__":
    unittest.main()
