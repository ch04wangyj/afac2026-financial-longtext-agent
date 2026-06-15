"""TokenUsage 聚合逻辑的单元测试。"""

import unittest

from agent.io.submission import summarize_usage
from agent.schemas import AnswerResult, TokenUsage


class TokenUsageTest(unittest.TestCase):
    def test_total_auto_fill(self):
        usage = TokenUsage(prompt_tokens=3, completion_tokens=2)
        self.assertEqual(usage.total_tokens, 5)

    def test_summarize_usage(self):
        results = [
            AnswerResult("q1", "A", 1.0, [], TokenUsage(3, 2)),
            AnswerResult("q2", "B", 1.0, [], TokenUsage(4, 1)),
        ]
        total = summarize_usage(results)
        self.assertEqual(total.prompt_tokens, 7)
        self.assertEqual(total.completion_tokens, 3)
        self.assertEqual(total.total_tokens, 10)


if __name__ == "__main__":
    unittest.main()
