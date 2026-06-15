"""原始文档解析器。

优先用轻量库读取 PDF/HTML/TXT，后续可在这里接入 MinerU、PaddleOCR 等更强解析器。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent.preprocess.normalization import normalize_text


@dataclass
class PageText:
    """单页解析结果，tables 字段为表格文本的轻量保留。"""

    page: int | None
    text: str
    tables: list[str]


def extract_pages(path: Path, domain: str = "") -> list[PageText]:
    """按文件类型分派解析逻辑，统一输出 PageText 列表。"""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [PageText(page=None, text=_read_text(path), tables=[])]
    if suffix == ".html":
        return [PageText(page=None, text=_read_html(path), tables=[])]
    if suffix == ".pdf":
        return _read_pdf(path, domain)
    raise ValueError(f"Unsupported raw file type: {path}")


def _read_text(path: Path) -> str:
    """读取普通文本，兼容 UTF-8 和常见中文编码。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return normalize_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def _read_html(path: Path) -> str:
    """提取 HTML 正文，去掉脚本和样式噪音。"""
    html = _read_text(path)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        body = soup.get_text("\n", strip=True)
        return normalize_text("\n".join(x for x in (title, body) if x))
    except Exception:
        text = re.sub(r"<[^>]+>", "\n", html)
        return normalize_text(text)


def _read_pdf(path: Path, domain: str) -> list[PageText]:
    """PDF 先走 PyMuPDF，失败时用 pdfplumber 兜底。"""
    try:
        return _read_pdf_with_fitz(path)
    except Exception as fitz_error:
        try:
            return _read_pdf_with_pdfplumber(path, include_tables=domain == "financial_contracts")
        except Exception as plumber_error:
            raise RuntimeError(
                f"PDF extraction failed for {path}. PyMuPDF error={fitz_error}; "
                f"pdfplumber error={plumber_error}"
            ) from plumber_error


def _read_pdf_with_fitz(path: Path) -> list[PageText]:
    """用 PyMuPDF 快速抽取电子版 PDF 页文本。"""
    import fitz

    pages: list[PageText] = []
    with fitz.open(path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            pages.append(PageText(page=idx, text=normalize_text(text), tables=[]))
    return pages


def _read_pdf_with_pdfplumber(path: Path, include_tables: bool) -> list[PageText]:
    """用 pdfplumber 兜底解析，并可额外提取合同表格。"""
    import pdfplumber

    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables: list[str] = []
            if include_tables:
                for table in page.extract_tables() or []:
                    rows = [" | ".join("" if cell is None else str(cell) for cell in row) for row in table]
                    tables.append("\n".join(rows))
            pages.append(PageText(page=idx, text=normalize_text(text), tables=tables))
    return pages
