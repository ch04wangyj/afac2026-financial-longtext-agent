"""带证据答案集的 exact-match 离线评估。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json

from agent.schemas import Question


_VALID_ANSWERS = {"A", "B", "C", "D"}


@dataclass(frozen=True)
class AnswerDevCase:
    """一条人工或官方核验的答案级开发样例。"""

    qid: str
    expected_answer: str
    answer_format: str
    domain: str
    provenance: str
    question_sha1: str = ""
    required_doc_ids: list[str] = field(default_factory=list)
    required_chunk_ids: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> "AnswerDevCase":
        return cls(
            qid=str(row["qid"]),
            expected_answer=str(row["expected_answer"]),
            answer_format=str(row["answer_format"]),
            domain=str(row["domain"]),
            provenance=str(row.get("provenance", "")),
            question_sha1=str(row.get("question_sha1", "")),
            required_doc_ids=[str(item) for item in row.get("required_doc_ids", [])],
            required_chunk_ids=[str(item) for item in row.get("required_chunk_ids", [])],
            notes=str(row.get("notes", "")),
        )


def evaluate_answer_devset(
    cases: list[AnswerDevCase],
    result_rows: list[dict],
    *,
    current_question_sha1: dict[str, str] | None = None,
) -> dict:
    """按比赛规则计算 exact-match，并检查关键文档与 chunk 是否进入证据链。"""
    by_qid = {str(row.get("qid", "")): row for row in result_rows if row.get("qid")}
    details: list[dict] = []
    domain_totals: dict[str, list[bool]] = defaultdict(list)
    correct_count = 0
    present_count = 0

    for case in cases:
        row = by_qid.get(case.qid)
        predicted = _normalize_answer(str((row or {}).get("answer", "")), case.answer_format)
        expected = _normalize_answer(case.expected_answer, case.answer_format)
        present = row is not None
        current_sha1 = (current_question_sha1 or {}).get(case.qid, "")
        question_version_ok = not case.question_sha1 or current_sha1 == case.question_sha1
        correct = present and question_version_ok and predicted == expected
        evidence_docs, evidence_chunks = _collect_evidence_scope(row or {})
        missing_docs = [doc_id for doc_id in case.required_doc_ids if doc_id not in evidence_docs]
        missing_chunks = [chunk_id for chunk_id in case.required_chunk_ids if chunk_id not in evidence_chunks]
        present_count += int(present)
        correct_count += int(correct)
        domain_totals[case.domain].append(correct)
        details.append(
            {
                "qid": case.qid,
                "domain": case.domain,
                "answer_format": case.answer_format,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "present": present,
                "correct": correct,
                "question_version_ok": question_version_ok,
                "expected_question_sha1": case.question_sha1,
                "current_question_sha1": current_sha1,
                "missing_doc_ids": missing_docs,
                "missing_chunk_ids": missing_chunks,
                "provenance": case.provenance,
            }
        )

    total = len(cases)
    case_map = cases_by_qid(cases)
    evidence_cases = [item for item in details if case_map[item["qid"]].required_chunk_ids]
    return {
        "total": total,
        "present": present_count,
        "correct": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "all_present": present_count == total,
        "all_question_versions_match": all(item["question_version_ok"] for item in details),
        "required_evidence_all_hit": sum(not item["missing_chunk_ids"] for item in evidence_cases),
        "required_evidence_case_count": len(evidence_cases),
        "by_domain": {
            domain: {
                "total": len(values),
                "correct": sum(values),
                "accuracy": sum(values) / len(values),
            }
            for domain, values in sorted(domain_totals.items())
        },
        "details": details,
    }


def cases_by_qid(cases: list[AnswerDevCase]) -> dict[str, AnswerDevCase]:
    return {case.qid: case for case in cases}


def question_sha1(question: Question) -> str:
    """对会影响答案的题面字段生成稳定指纹。"""
    payload = {
        "qid": question.qid,
        "question": question.question,
        "options": dict(sorted(question.options.items())),
        "answer_format": question.answer_format,
        "doc_ids": list(question.doc_ids),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _normalize_answer(answer: str, answer_format: str) -> str:
    letters = [letter for letter in answer.upper() if letter in _VALID_ANSWERS]
    if answer_format == "multi":
        return "".join(sorted(set(letters)))
    return letters[0] if letters else ""


def _collect_evidence_scope(row: dict) -> tuple[set[str], set[str]]:
    docs: set[str] = set()
    chunks: set[str] = set()
    for item in row.get("evidence", []) or []:
        if item.get("doc_id"):
            docs.add(str(item["doc_id"]))
        if item.get("chunk_id"):
            chunks.add(str(item["chunk_id"]))
        parent_chunk_id = (item.get("metadata") or {}).get("parent_chunk_id")
        if parent_chunk_id:
            chunks.add(str(parent_chunk_id))
    claim_runs = ((row.get("metadata") or {}).get("claim_runs") or {})
    for run in claim_runs.values():
        docs.update(str(item) for item in run.get("evidence_doc_ids", []) if item)
        chunks.update(str(item) for item in run.get("evidence_chunk_ids", []) if item)
    return docs, chunks
