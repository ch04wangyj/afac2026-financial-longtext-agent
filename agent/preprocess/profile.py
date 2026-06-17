"""预处理画像与样本导出辅助函数。"""

from __future__ import annotations

from pathlib import Path


def sample_output_dir(domain: str, doc_id: str, outputs_root: Path | str = Path("outputs")) -> Path:
    """返回单个 Docling 样本的输出目录。"""
    return Path(outputs_root) / "docling_samples" / domain / doc_id
