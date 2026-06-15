import tempfile
"""doc_id 到原始文档路径映射的单元测试。"""

import unittest
from pathlib import Path

from agent.data.doc_registry import DocRegistry


class DocRegistryTest(unittest.TestCase):
    def test_resolve_five_domains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "insurance": "1.pdf",
                "financial_contracts": "text01.pdf",
                "financial_reports": "annual_byd_2024_report.PDF",
                "research": "pack2_text01.pdf",
                "regulatory/txt": "strict_v3_008_办法.txt",
            }
            for domain_path, name in files.items():
                directory = root / domain_path
                directory.mkdir(parents=True, exist_ok=True)
                (directory / name).write_text("x", encoding="utf-8")

            registry = DocRegistry(root)
            self.assertEqual(registry.resolve("insurance", "1").name, "1.pdf")
            self.assertEqual(registry.resolve("financial_contracts", "text01").name, "text01.pdf")
            self.assertEqual(registry.resolve("financial_reports", "annual_byd_2024_report").name, "annual_byd_2024_report.PDF")
            self.assertEqual(registry.resolve("research", "pack2_text01").name, "pack2_text01.pdf")
            self.assertEqual(registry.resolve("regulatory", "strict_v3_008").name, "strict_v3_008_办法.txt")


if __name__ == "__main__":
    unittest.main()
