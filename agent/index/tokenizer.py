"""中文金融文本的词法 tokenizer。

默认 mixed 模式同时使用中文 char n-gram 和 jieba 分词，兼顾精确术语和分词失败时
的鲁棒性；全程不使用 embedding。
"""

from __future__ import annotations

import re

from agent.preprocess.normalization import compact_for_search
from agent.schemas import Chunk


TERM_RE = re.compile(r"[A-Za-z0-9_.%-]+|[\u4e00-\u9fff]+")


TokenizerMode = str


def tokenize(text: str, use_jieba: bool = True, mode: TokenizerMode = "mixed") -> list[str]:
    """把查询或正文切成 BM25 可用 token。"""
    text = compact_for_search(text).lower()
    tokens: list[str] = []

    for match in TERM_RE.finditer(text):
        term = match.group(0)
        if _is_chinese(term):
            term_tokens: list[str] = []
            # char n-gram 对中文长术语更稳，避免 jieba 词典缺项导致召回失败。
            if mode in {"mixed", "char"}:
                term_tokens.extend(char_ngrams(term, min_n=2, max_n=4))
            if mode in {"mixed", "word"} and use_jieba:
                term_tokens.extend(_jieba_tokens(term))
            if mode == "word" and not term_tokens:
                term_tokens.append(term)
            tokens.extend(term_tokens)
        else:
            tokens.append(term)

    return [tok for tok in tokens if tok]


def tokenize_chunk(chunk: Chunk, mode: TokenizerMode = "mixed") -> list[str]:
    """索引 chunk 时把标题、章节、条款、表格、数字和日期都纳入检索字段。"""
    fields = [
        chunk.metadata.get("title", ""),
        chunk.section,
        chunk.clause_id,
        chunk.text,
        " ".join(chunk.tables),
        " ".join(chunk.numbers),
        " ".join(chunk.dates),
    ]
    return tokenize(" ".join(fields), mode=mode)


def char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    """生成字符 n-gram，适合中文法规名、产品名和财务指标召回。"""
    output: list[str] = []
    length = len(text)
    for n in range(min_n, max_n + 1):
        if length < n:
            continue
        output.extend(text[i : i + n] for i in range(length - n + 1))
    if not output and text:
        output.append(text)
    return output


def _jieba_tokens(text: str) -> list[str]:
    """可选使用 jieba 分词；缺依赖时自动降级为空列表。"""
    try:
        import jieba

        return [tok.strip().lower() for tok in jieba.cut(text) if len(tok.strip()) >= 2]
    except Exception:
        return []


def _is_chinese(text: str) -> bool:
    """判断一个 term 是否全部为中文字符。"""
    return all("\u4e00" <= ch <= "\u9fff" for ch in text)
