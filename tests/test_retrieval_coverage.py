from agent.schemas import RetrievalResult

from agent.retrieve.coverage import (
    EvidenceCoverageReport,
    assess_doc_coverage,
    retrieve_missing_doc_evidence,
)


def _result(doc_id: str, chunk_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"{doc_id}:1",
        doc_id=doc_id,
        domain="financial_reports",
        score=1.0,
        source="test",
        query="sample query",
        evidence_text="sample",
        metadata={},
    )


def test_assess_doc_coverage_reports_missing_doc_ids():
    report = assess_doc_coverage(required_doc_ids=["d1", "d2"], evidence=[_result("d1")])

    assert isinstance(report, EvidenceCoverageReport)
    assert report.covered_doc_ids == ["d1"]
    assert report.missing_doc_ids == ["d2"]
    assert report.ok is False


def test_assess_doc_coverage_ok_when_all_docs_present():
    report = assess_doc_coverage(required_doc_ids=["d1", "d2"], evidence=[_result("d2"), _result("d1")])

    assert report.missing_doc_ids == []
    assert report.ok is True


def test_retrieve_missing_doc_evidence_queries_each_missing_doc():
    class FakeIndex:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int, set[str] | None, str]] = []

        def search(self, query, top_k, filter_doc_ids=None, source=""):
            self.calls.append((query, top_k, filter_doc_ids, source))
            doc_id = next(iter(filter_doc_ids))
            return [_result(doc_id)]

    index = FakeIndex()

    evidence = retrieve_missing_doc_evidence(
        index=index,
        query="发行规模 主体评级",
        missing_doc_ids=["d2", "d3"],
        top_k=2,
    )

    assert [item.doc_id for item in evidence] == ["d2", "d3"]
    assert index.calls[0][2] == {"d2"}
    assert index.calls[1][2] == {"d3"}
    assert all(call[3] == "coverage_missing_doc" for call in index.calls)
