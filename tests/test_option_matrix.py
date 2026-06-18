from agent.reasoning.prompts import build_option_evidence_judgement_messages
from agent.reasoning.option_matrix import OptionVerdict, parse_option_verdict, synthesize_answer
from agent.schemas import Question, RetrievalResult


def _question() -> Question:
    return Question(
        qid="q_option_matrix",
        domain="financial_reports",
        split="A",
        question="根据财报披露，以下哪些说法正确？",
        options={
            "A": "2025 年收入增长",
            "B": "2025 年收入下降",
            "C": "研发费用减少",
        },
        answer_format="multi",
        doc_ids=["doc-1"],
    )



def _evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="doc-1:1",
            doc_id="doc-1",
            domain="financial_reports",
            score=1.0,
            source="test",
            query="收入增长",
            evidence_text="2025 年公司营业收入同比增长 12%。",
            metadata={"title": "2025 年年报", "page": 3},
        ),
        RetrievalResult(
            chunk_id="doc-1:2",
            doc_id="doc-1",
            domain="financial_reports",
            score=0.8,
            source="test",
            query="研发费用",
            evidence_text="研发费用同比增加 5%。",
            metadata={"title": "2025 年年报", "page": 10},
        ),
    ]



def test_synthesize_multi_answer_from_true_verdicts():
    verdicts = {
        "A": OptionVerdict(option="A", verdict=True, confidence=0.9),
        "B": OptionVerdict(option="B", verdict=False, confidence=0.8),
        "C": OptionVerdict(option="C", verdict=True, confidence=0.7),
    }
    assert synthesize_answer(verdicts, answer_format="multi") == "AC"



def test_synthesize_mcq_picks_highest_true_confidence():
    verdicts = {
        "A": OptionVerdict(option="A", verdict=True, confidence=0.6),
        "B": OptionVerdict(option="B", verdict=True, confidence=0.9),
    }
    assert synthesize_answer(verdicts, answer_format="mcq") == "B"



def test_build_option_evidence_judgement_prompt_requires_support_refute_ids():
    messages = build_option_evidence_judgement_messages(_question(), "A", "2025 年收入增长", _evidence())
    text = messages[-1]["content"]
    assert "support_evidence" in text
    assert "refute_evidence" in text
    assert "insufficient" in text
    assert "只输出 JSON" in text



def test_parse_support_relation_as_true_verdict():
    verdict = parse_option_verdict(
        '{"option":"A","relation":"support","confidence":0.8,"support_evidence":["[1]"],"reason":"原文支持"}',
        "A",
    )
    assert verdict.verdict is True
    assert verdict.support_evidence == ["[1]"]



def test_parse_insufficient_relation_as_none():
    verdict = parse_option_verdict('{"option":"B","relation":"insufficient","confidence":0.2}', "B")
    assert verdict.verdict is None
