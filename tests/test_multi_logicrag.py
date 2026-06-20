from agent.reasoning.multi_logicrag import assemble_multi_logicrag_answer, should_expand_uncertain_option
from agent.reasoning.option_matrix import OptionVerdict


def test_assemble_multi_logicrag_answer_uses_only_supported_options():
    verdicts = {
        "A": OptionVerdict(option="A", verdict=True, confidence=0.95, support_evidence=["[1]"], reason="supported"),
        "B": OptionVerdict(option="B", verdict=True, confidence=0.90, support_evidence=["[2]"], reason="supported"),
        "C": OptionVerdict(option="C", verdict=None, confidence=0.20, reason="insufficient"),
        "D": OptionVerdict(option="D", verdict=False, confidence=0.85, refute_evidence=["[3]"], reason="refuted"),
    }

    answer = assemble_multi_logicrag_answer(verdicts)

    assert answer == "AB"


def test_should_expand_uncertain_option_for_insufficient_verdict():
    verdict = OptionVerdict(option="A", verdict=None, confidence=0.40, reason="insufficient")

    assert should_expand_uncertain_option(verdict, coverage={"missing_doc_ids": []}, threshold=0.7) is True


def test_should_expand_uncertain_option_for_low_confidence_without_evidence():
    verdict = OptionVerdict(option="B", verdict=True, confidence=0.55, support_evidence=[], refute_evidence=[], reason="weak")

    assert should_expand_uncertain_option(verdict, coverage={"missing_doc_ids": []}, threshold=0.7) is True


def test_should_expand_uncertain_option_for_missing_doc_coverage():
    verdict = OptionVerdict(option="C", verdict=False, confidence=0.91, refute_evidence=["[1]"], reason="refuted")

    assert should_expand_uncertain_option(verdict, coverage={"missing_doc_ids": ["doc2"]}, threshold=0.7) is True


def test_should_not_expand_confident_supported_option_with_evidence():
    verdict = OptionVerdict(option="D", verdict=True, confidence=0.95, support_evidence=["[2]"], reason="supported")

    assert should_expand_uncertain_option(verdict, coverage={"missing_doc_ids": []}, threshold=0.7) is False
