"""doc_id 到原始文件路径的映射器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent.schemas import Document, Question


SUPPORTED_RAW_EXTS = {".pdf", ".PDF", ".html", ".txt"}


@dataclass
class DocRegistry:
    """扫描 raw 目录并按领域解析题目中的 doc_id。"""

    raw_root: Path

    def __post_init__(self) -> None:
        """初始化缓存，避免重复扫描大目录。"""
        self.raw_root = Path(self.raw_root)
        self._files_by_domain: dict[str, list[Path]] = {}
        self._stem_index: dict[str, dict[str, Path]] = {}

    def resolve(self, domain: str, doc_id: str) -> Path:
        """解析单个 doc_id，按精确、前缀、包含三档匹配。"""
        domain_files = self._domain_files(domain)
        exact = self._stem_index[domain].get(doc_id.lower())
        if exact:
            return exact

        candidates = [path for path in domain_files if path.stem.lower() == doc_id.lower()]
        if candidates:
            return candidates[0]

        starts = [path for path in domain_files if path.stem.lower().startswith(doc_id.lower())]
        if starts:
            return sorted(starts, key=lambda p: len(p.name))[0]

        contains = [path for path in domain_files if doc_id.lower() in path.stem.lower()]
        if contains:
            return sorted(contains, key=lambda p: len(p.name))[0]

        raise FileNotFoundError(f"Cannot resolve doc_id={doc_id!r} in domain={domain!r}")

    def build_documents_for_questions(self, questions: list[Question]) -> list[Document]:
        """从题目集合生成待解析文档列表，并对同一文档去重。"""
        seen: set[tuple[str, str]] = set()
        docs: list[Document] = []
        for question in questions:
            for doc_id in question.doc_ids:
                key = (question.domain, doc_id)
                if key in seen:
                    continue
                path = self.resolve(question.domain, doc_id)
                docs.append(
                    Document(
                        doc_id=doc_id,
                        domain=question.domain,
                        title=path.stem,
                        path=str(path),
                        metadata={"source_suffix": path.suffix, "question_split": question.split},
                    )
                )
                seen.add(key)
        return docs

    def _domain_files(self, domain: str) -> list[Path]:
        """懒加载某个领域下支持的原始文件。"""
        if domain in self._files_by_domain:
            return self._files_by_domain[domain]

        domain_root = self.raw_root / domain
        if not domain_root.exists():
            raise FileNotFoundError(f"Raw domain directory not found: {domain_root}")

        files = [
            path
            for path in domain_root.rglob("*")
            if path.is_file() and path.suffix in SUPPORTED_RAW_EXTS
        ]
        self._files_by_domain[domain] = files
        self._stem_index[domain] = {path.stem.lower(): path for path in files}
        return files
