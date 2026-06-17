"""预处理画像与样本导出辅助函数测试。"""

import unittest

from agent.preprocess.profile import sample_output_dir


class PreprocessProfileTest(unittest.TestCase):
    def test_sample_output_dir_uses_domain_and_doc_id(self):
        path = sample_output_dir("financial_reports", "annual_cscec_2024_report")
        self.assertEqual(
            path.as_posix(),
            "outputs/docling_samples/financial_reports/annual_cscec_2024_report",
        )


if __name__ == "__main__":
    unittest.main()
