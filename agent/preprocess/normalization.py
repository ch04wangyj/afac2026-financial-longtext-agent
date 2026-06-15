"""文本清洗工具，统一 PDF/HTML/TXT 的空白和全角字符。"""

from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """做 NFKC 归一化并压缩多余空白。"""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u3000", " ")
    text = text.replace("\xa0", " ")
    text = WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def compact_for_search(text: str) -> str:
    """为检索构造单行文本，避免换行影响 tokenizer。"""
    return re.sub(r"\s+", " ", normalize_text(text))
