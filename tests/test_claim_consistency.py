from agent.evaluation.claim_consistency import (
    build_claim_records,
    claim_similarity,
    find_claim_conflicts,
    normalize_claim,
    numeric_signature,
    qualifier_signature,
)
from agent.schemas import AnswerResult, Question, TokenUsage


def _answer(qid: str, answer: str) -> AnswerResult:
    return AnswerResult(
        qid=qid,
        answer=answer,
        confidence=1.0,
        evidence=[],
        token_usage=TokenUsage(),
    )


def _question(qid: str, text: str, *, answer_format: str = "multi") -> Question:
    return Question(
        qid=qid,
        domain="regulatory",
        split="A",
        question=text if answer_format == "tf" else "测试题",
        options={"A": text, "B": "无关选项"},
        answer_format=answer_format,
        doc_ids=["doc-1"],
    )


def test_normalize_claim_and_numeric_signature_are_stable():
    assert normalize_claim("截至 2025 年，累计同比增速为 4.09%。") == "截至2025年累计同比为4.09%"
    assert numeric_signature("从 1.56 倍提升至 4.09 倍") == ("1.56倍", "4.09倍")
    assert numeric_signature("至少提前三十个自然日") == ("三十个自然日",)
    assert qualifier_signature("收入同比下降 3.6%") != qualifier_signature("收入同比增长 3.6%")


def test_build_claim_records_maps_tf_answer_to_claim_truth():
    question = _question(
        "q1",
        "判断以下陈述是否正确：指标由 1.56 倍升至 4.09 倍。",
        answer_format="tf",
    )
    records = build_claim_records([question], [_answer("q1", "B")])

    assert len(records) == 1
    assert records[0].option == "TF"
    assert records[0].selected is False
    assert records[0].text == "指标由 1.56 倍升至 4.09 倍。"


def test_find_claim_conflicts_requires_shared_document_and_matching_numbers():
    questions = [
        _question("q1", "重大差异应在发现之日起30个工作日内提交差异报告"),
        _question("q2", "发现重大差异后，应当在30个工作日内提交差异报告"),
        _question("q3", "发现重大差异后，应当在60个工作日内提交差异报告"),
    ]
    answers = [_answer("q1", "A"), _answer("q2", "B"), _answer("q3", "B")]
    records = build_claim_records(questions, answers)
    conflicts = find_claim_conflicts(records, min_similarity=0.65)

    pairs = {(item.left.qid, item.right.qid) for item in conflicts}
    assert ("q1", "q2") in pairs
    assert ("q1", "q3") not in pairs


def test_find_claim_conflicts_rejects_different_document_sets_by_default():
    left = _question("q1", "2025年市场规模接近2500亿元")
    right = _question("q2", "2025年市场规模预计接近2500亿元")
    right.doc_ids = ["doc-1", "doc-2"]
    records = build_claim_records([left, right], [_answer("q1", "A"), _answer("q2", "B")])

    assert find_claim_conflicts(records, min_similarity=0.7) == []
    assert find_claim_conflicts(
        records,
        min_similarity=0.7,
        require_same_doc_set=False,
    )


def test_support_contract_allows_safe_cross_question_document_overlap():
    left = _question("q1", "重大差异应在30个工作日内提交差异报告")
    left.doc_ids = ["rule-a", "rule-b"]
    right = _question("q2", "重大差异应当在30个工作日内提交差异报告")
    right.doc_ids = ["rule-a", "rule-c"]
    support = [
        _support_result("q1", "A", ["rule-a"]),
        _support_result("q2", "A", ["rule-a"]),
    ]
    records = build_claim_records(
        [left, right],
        [_answer("q1", "A"), _answer("q2", "B")],
        support_results=support,
    )

    conflicts = find_claim_conflicts(records, min_similarity=0.7)
    assert conflicts
    assert conflicts[0].shared_doc_ids == ("rule-a",)


def test_claim_similarity_handles_punctuation_only_rewrite():
    left = normalize_claim("服务零售累计同比增速高于商品零售。")
    right = normalize_claim("服务零售累计同比，高于商品零售！")
    assert claim_similarity(left, right) > 0.9


def _support_result(qid: str, option: str, doc_ids: list[str]) -> AnswerResult:
    row = _answer(qid, "A")
    row.metadata = {
        "retrieval_report": {
            "evidence_contracts": {
                option: {
                    "predicate_doc_ids": doc_ids,
                }
            }
        }
    }
    return row
