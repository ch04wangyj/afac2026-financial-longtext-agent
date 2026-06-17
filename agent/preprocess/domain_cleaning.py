"""按领域执行专用文本清洗。"""

from __future__ import annotations

import re

from agent.preprocess.normalization import normalize_text

TOC_LINE_RE = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+节\s+.*|[一二三四五六七八九十]+、.*|\d+(?:\.\d+)+.*)$")
CHECKBOX_NOISE_RE = re.compile(r"[□■✓✔]")
IMAGE_PLACEHOLDER_RE = re.compile(r"^<!--\s*image\s*-->$", re.IGNORECASE)


def clean_domain_text(domain: str, text: str, rules: list[str] | None = None) -> str:
    """根据领域规则清洗文本，同时保留与检索相关的正文内容。"""
    cleaned = normalize_text(text)
    rules = list(rules or [])

    if "drop_toc_blocks" in rules:
        cleaned = _drop_toc_lines(cleaned)
    if "downweight_template_sections" in rules:
        cleaned = _drop_template_noise(domain, cleaned)
    if domain == "financial_reports":
        cleaned = _drop_checkbox_noise(cleaned)
    return normalize_text(cleaned)


def _drop_toc_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "目录":
            continue
        if TOC_LINE_RE.match(stripped) and ("...." in stripped or "..." in stripped or stripped.startswith(("第一节", "第二节", "第三节", "第四节"))):
            continue
        lines.append(line)
    return "\n".join(lines)


def _drop_template_noise(domain: str, text: str) -> str:
    drop_phrases = {
        "financial_contracts": ["公司声明", "重大事项提示", "重大风险提示", "释 义", "释义"],
        "financial_reports": ["重要提示", "本年度报告载有若干涉及本公司未来计划"],
    }
    blocked = tuple(drop_phrases.get(domain, []))
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if blocked and stripped.startswith(blocked):
            continue
        if IMAGE_PLACEHOLDER_RE.match(stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def _drop_checkbox_noise(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = CHECKBOX_NOISE_RE.sub("", line).strip()
        if stripped in {"适用 不适用", "是 否", "不适用", "适用"}:
            continue
        lines.append(stripped or line)
    return "\n".join(lines)
