from agent.evaluation.answer_devset import AnswerDevCase, evaluate_answer_devset, question_sha1
from agent.schemas import Question


def _case(qid: str, expected: str = "AC") -> AnswerDevCase:
    return AnswerDevCase(
        qid=qid,
        expected_answer=expected,
        answer_format="multi",
        domain="regulatory",
        provenance="test",
        required_doc_ids=["doc1"],
        required_chunk_ids=["c1"],
    )


def test_devset_uses_multi_exact_match_and_collects_claim_evidence():
    report = evaluate_answer_devset(
        [_case("q1")],
        [
            {
                "qid": "q1",
                "answer": "CA",
                "evidence": [],
                "metadata": {
                    "claim_runs": {
                        "A": {"evidence_doc_ids": ["doc1"], "evidence_chunk_ids": ["c1"]}
                    }
                },
            }
        ],
    )

    assert report["accuracy"] == 1.0
    assert report["required_evidence_all_hit"] == 1
    assert report["details"][0]["missing_doc_ids"] == []


def test_devset_counts_missing_result_as_incorrect():
    report = evaluate_answer_devset([_case("q1")], [])

    assert report["present"] == 0
    assert report["correct"] == 0
    assert report["accuracy"] == 0.0
    assert report["all_present"] is False


def test_question_fingerprint_detects_stale_dev_label():
    question = Question(
        qid="q1",
        domain="regulatory",
        split="A",
        question="原题",
        options={"A": "正确", "B": "错误"},
        answer_format="mcq",
        doc_ids=["doc1"],
    )
    case = AnswerDevCase(
        qid="q1",
        expected_answer="A",
        answer_format="mcq",
        domain="regulatory",
        provenance="test",
        question_sha1=question_sha1(question),
    )

    report = evaluate_answer_devset(
        [case],
        [{"qid": "q1", "answer": "A"}],
        current_question_sha1={"q1": "stale"},
    )

    assert report["correct"] == 0
    assert report["all_question_versions_match"] is False
