"""Helpers for multi-select enhanced LogicRAG."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.reasoning.option_matrix import OptionVerdict, synthesize_answer
from agent.retrieve.structured_queries import extract_query_entities
from agent.retrieve.targets import analyze_evidence_sufficiency, build_retrieval_target
from agent.schemas import Question, RetrievalResult, TokenUsage


@dataclass
class MultiOptionRun:
    option_key: str
    option_text: str
    verdict: OptionVerdict
    evidence: list[RetrievalResult] = field(default_factory=list)
    retried: bool = False
    retry_reason: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "option": self.option_key,
            "verdict": self.verdict.to_dict(),
            "retried": self.retried,
            "retry_reason": self.retry_reason,
            "token_usage": self.token_usage.to_dict(),
            "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in self.evidence)),
            "evidence_chunk_ids": [item.chunk_id for item in self.evidence],
            "metadata": dict(self.metadata),
        }



def build_multi_option_question(question: Question, option_key: str, option_text: str) -> Question:
    return Question(
        qid=f"{question.qid}:{option_key}",
        domain=question.domain,
        split=question.split,
        question=f"{question.question}\n判断选项{option_key}是否正确：{option_text}",
        options={option_key: option_text},
        answer_format="tf",
        type=question.type,
        doc_ids=list(question.doc_ids),
    )



def assemble_multi_logicrag_answer(verdicts: dict[str, OptionVerdict]) -> str:
    """Assemble final multi answer from option verdicts only."""
    return synthesize_answer(verdicts, "multi")



def should_expand_uncertain_option(
    verdict: OptionVerdict,
    *,
    coverage: dict | None,
    threshold: float,
) -> bool:
    """Every uncertain option must receive one wider retrieval pass."""
    coverage = coverage or {}
    missing_doc_ids = list(coverage.get("missing_doc_ids") or [])
    no_evidence_refs = not verdict.support_evidence and not verdict.refute_evidence
    return (
        verdict.verdict is None
        or float(verdict.confidence or 0.0) < float(threshold)
        or no_evidence_refs
        or bool(missing_doc_ids)
    )



def build_retry_queries(question: Question, option_key: str, option_text: str, base_queries: list[str]) -> list[str]:
    """Expand query set for an uncertain option without rerunning the whole question."""
    stem_entities = extract_query_entities(question.question)
    option_entities = extract_query_entities(option_text)
    numbers = extract_numbers(f"{question.question} {option_text}")
    dates = extract_dates(f"{question.question} {option_text}")
    queries = [*base_queries]
    queries.extend(
        [
            f"{question.question} {option_key} {option_text}".strip(),
            f"{question.question} {option_text}".strip(),
            f"{option_text} {question.question}".strip(),
        ]
    )
    if stem_entities or option_entities:
        queries.append(" ".join([*stem_entities[:8], *option_entities[:8]]))
    if numbers:
        queries.append(f"{question.question} {' '.join(numbers)}")
    if dates:
        queries.append(f"{question.question} {' '.join(dates)}")
    return _dedupe(queries)



def build_gap_aware_retry_query(question: Question, option_key: str, option_text: str, evidence: list[RetrievalResult]) -> str:
    target = build_retrieval_target(question, f"{option_key} {option_text}")
    report = analyze_evidence_sufficiency(target, [item.evidence_text for item in evidence])
    missing_parts = [
        *report.get("missing_entities", [])[:2],
        *report.get("missing_numbers", [])[:2],
        *report.get("missing_dates", [])[:2],
        *report.get("missing_must_terms", [])[:2],
    ]
    gap_suffix = " ".join(dict.fromkeys(part for part in missing_parts if part))
    return " ".join(part for part in [question.question, option_key, option_text, gap_suffix] if part).strip()



def merge_unique_evidence(*groups: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: list[RetrievalResult] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.chunk_id in seen:
                continue
            merged.append(item)
            seen.add(item.chunk_id)
    return merged



def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join((item or "").split())
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
