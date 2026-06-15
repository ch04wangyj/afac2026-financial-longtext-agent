"""题目读取工具。"""

from __future__ import annotations

from pathlib import Path

from agent.io.jsonl import read_json
from agent.schemas import Question


def load_questions(questions_root: Path, domains: list[str] | None = None) -> list[Question]:
    """读取 group_a 下各领域题目 JSON，可按领域过滤。"""
    files = sorted(questions_root.glob("*_questions.json"))
    if domains:
        allowed = set(domains)
        files = [path for path in files if path.name.replace("_questions.json", "") in allowed]

    questions: list[Question] = []
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Question file must contain a list: {path}")
        questions.extend(Question.from_dict(item) for item in payload)
    return questions


def unique_doc_ids(questions: list[Question]) -> dict[str, set[str]]:
    """按领域汇总题目引用过的 doc_id，用于最小化预处理范围。"""
    mapping: dict[str, set[str]] = {}
    for question in questions:
        mapping.setdefault(question.domain, set()).update(question.doc_ids)
    return mapping
