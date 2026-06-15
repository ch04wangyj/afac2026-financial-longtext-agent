"""不同 tokenizer 模式的单元测试。"""

import unittest

from agent.index.tokenizer import tokenize


class TokenizerModesTest(unittest.TestCase):
    def test_char_mode_has_ngrams(self):
        tokens = tokenize("身故保险金", mode="char")
        self.assertIn("身故", tokens)
        self.assertIn("保险金", tokens)

    def test_word_mode_keeps_signal(self):
        tokens = tokenize("身故保险金", mode="word")
        self.assertTrue(tokens)


if __name__ == "__main__":
    unittest.main()
