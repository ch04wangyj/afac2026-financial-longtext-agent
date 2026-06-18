"""Evidence coverage helpers for A-board retrieval quality gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.schemas import RetrievalResult


@dataclass(frozen=True)
class EvidenceCoverageReport:
    required_doc_ids: list[str] = field(default_factory=list)
    covered_doc_ids: list[str] = field(default_factory=list)
    missing_doc_ids: list[str] = field(default_factory=list)
    option_missing: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_doc_ids and not any(self.option_missing.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_doc_ids": list(self.required_doc_ids),
            "covered_doc_ids": list(self.covered_doc_ids),
            "missing_doc_ids": list(self.missing_doc_ids),
            "option_missing": {key: list(value) for key, value in self.option_missing.items()},
            "notes": list(self.notes),
            "ok": self.ok,
        }


def assess_doc_coverage(
    required_doc_ids: list[str],
    evidence: list[RetrievalResult],
) -> EvidenceCoverageReport:
    required = list(dict.fromkeys(required_doc_ids or []))
    evidence_doc_ids = {item.doc_id for item in evidence}
    covered = [doc_id for doc_id in required if doc_id in evidence_doc_ids]
    missing = [doc_id for doc_id in required if doc_id not in evidence_doc_ids]
    return EvidenceCoverageReport(
        required_doc_ids=required,
        covered_doc_ids=covered,
        missing_doc_ids=missing,
    )


def retrieve_missing_doc_evidence(
    index,
    query: str,
    missing_doc_ids: list[str],
    top_k: int = 6,
) -> list[RetrievalResult]:
    output: list[RetrievalResult] = []
    seen: set[str] = set()
    for doc_id in dict.fromkeys(missing_doc_ids or []):
        results = index.search(
            query=query,
            top_k=top_k,
            filter_doc_ids={doc_id},
            source="coverage_missing_doc",
        )
        for item in results:
            if item.chunk_id not in seen:
                output.append(item)
                seen.add(item.chunk_id)
    return output
