"""答案解析器的单元测试。"""

import unittest

from agent.reasoning.answer_parser import parse_answer, parse_verdict
from agent.reasoning.solver import _extract_confidence


class AnswerParserTest(unittest.TestCase):
    def test_json_single(self):
        self.assertEqual(parse_answer('{"answer":"C","confidence":0.8}', "mcq"), "C")

    def test_truncated_json_answer_field(self):
        self.assertEqual(parse_answer('{"answer": "C", "confidence": 0.95, "reason": "截断', "mcq"), "C")

    def test_label_single(self):
        self.assertEqual(parse_answer("最终答案：D，因为证据支持。", "tf"), "D")

    def test_multi_sorted_unique(self):
        self.assertEqual(parse_answer("ANSWER: C, A, C", "multi"), "AC")

    def test_illegal_empty(self):
        self.assertEqual(parse_answer("没有明确答案", "mcq"), "")

    def test_english_field_names_do_not_create_fallback_answer(self):
        self.assertEqual(parse_answer('{"confidence": 0.5, "reason": "no answer yet"', "mcq"), "")

    def test_confidence_from_fenced_json(self):
        text = '```json\n{"answer":"A","confidence":0.9,"reason":"x"}\n```'
        self.assertEqual(_extract_confidence(text), 0.9)

    def test_confidence_from_truncated_json_prefix(self):
        text = '{"verdict": false, "confidence": 0.95, "reason": "x"'
        self.assertEqual(_extract_confidence(text), 0.95)

    def test_parse_verdict_from_json(self):
        self.assertIs(parse_verdict('{"verdict": true, "confidence": 0.8}'), True)
        self.assertIs(parse_verdict('{"verdict": "false", "confidence": 0.8}'), False)

    def test_parse_verdict_from_truncated_json_prefix(self):
        text = '{"verdict": false, "confidence": 0.95, "reason": "后文出现 verdict=true 只是反思'
        self.assertIs(parse_verdict(text), False)


if __name__ == "__main__":
    unittest.main()
