"""模型答案解析器。

兼容 JSON、显式“答案: A”和自由文本中的 A-D 字母，确保最终提交格式合法。
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.schemas import AnswerFormat


VALID = {"A", "B", "C", "D"}
ANSWER_FIELD_RE = re.compile(
    r"(?i)[\"']?(?:answer|答案|最终答案|final_answer)[\"']?\s*[:：]\s*[\"']?([A-D](?:\s*,?\s*[A-D])*)"
)
VERDICT_FIELD_RE = re.compile(r"(?i)[\"']?(?:verdict|is_correct|correct|判断)[\"']?\s*[:：]\s*[\"']?(true|false|yes|no|正确|错误|对|错)")
TRUE_WORDS = {"true", "yes", "正确", "对", "成立", "支持", "是"}
FALSE_WORDS = {"false", "no", "错误", "错", "不成立", "不支持", "否"}


def parse_answer(text: str, answer_format: AnswerFormat) -> str:
    """根据题型抽取答案；多选会去重并按字母排序。"""
    candidates = _extract_candidates(text)
    if answer_format == "multi":
        letters = sorted(set(letter for letter in candidates if letter in VALID))
        return "".join(letters)
    for letter in candidates:
        if letter in VALID:
            return letter
    return ""


def parse_verdict(text: str) -> bool | None:
    """解析逐选项判断结果，返回 True/False；无法确定时返回 None。"""
    obj = extract_json_object(text)
    if obj:
        value = obj.get("verdict") or obj.get("is_correct") or obj.get("correct") or obj.get("判断")
        parsed = _parse_bool_value(value)
        if parsed is not None:
            return parsed
    field_match = VERDICT_FIELD_RE.search(text or "")
    if field_match:
        return _parse_bool_value(field_match.group(1))
    lowered = (text or "").strip().lower()
    # 否定词优先，避免“不正确”被“正确”误判，也避免截断文本中后续反思污染结果。
    if any(word in lowered for word in FALSE_WORDS):
        return False
    if any(word in lowered for word in TRUE_WORDS):
        return True
    return None


def _extract_candidates(text: str) -> list[str]:
    """按 JSON、答案字段、全文扫描三层策略抽取候选字母。"""
    text = text or ""
    json_answer = _try_json_answer(text)
    if json_answer:
        return _letters(json_answer)

    match = ANSWER_FIELD_RE.search(text)
    if match:
        return _letters(match.group(1))

    return _standalone_letters(text)


def _try_json_answer(text: str) -> str:
    """尝试从模型输出中解析 answer 字段。"""
    obj = extract_json_object(text)
    if not obj:
        return ""
    value = obj.get("answer") or obj.get("答案") or obj.get("final_answer")
    return str(value) if value is not None else ""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从裸 JSON、fenced JSON 或混合文本中提取第一个 JSON 对象。"""
    stripped = text.strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fenced:
        stripped = fenced.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _letters(text: str) -> list[str]:
    """提取 A-D 字母并统一大写。"""
    return [letter.upper() for letter in re.findall(r"[A-D]", text.upper())]


def _standalone_letters(text: str) -> list[str]:
    """只提取独立答案字母，避免从 answer/confidence 等英文单词里误取 A-D。"""
    return [letter.upper() for letter in re.findall(r"(?<![A-Za-z])[A-D](?![A-Za-z])", text.upper())]


def _parse_bool_value(value) -> bool | None:
    """把 JSON 字段中的布尔/字符串判断统一为 Python bool。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    return None
