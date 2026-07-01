import pytest

from agent.evaluation.label_evidence import (
    LabelEvidence,
    LabelEvidenceLayer,
    build_benchmark_assignment,
)


def test_semantic_evidence_cannot_fix_hidden_benchmark_label():
    evidence = [
        LabelEvidence(
            qid="q1",
            answer="A",
            layer=LabelEvidenceLayer.SOURCE_SEMANTIC,
            confirmed=True,
            source="document.pdf:12",
        )
    ]

    with pytest.raises(ValueError, match="不能固定比赛隐藏标签"):
        build_benchmark_assignment(evidence)


def test_official_and_mathematically_forced_labels_can_be_fixed():
    evidence = [
        LabelEvidence(
            qid="q1",
            answer="A",
            layer=LabelEvidenceLayer.OFFICIAL_LABEL,
            confirmed=True,
            source="official",
        ),
        LabelEvidence(
            qid="q2",
            answer="B",
            layer=LabelEvidenceLayer.LEADERBOARD_FORCED,
            confirmed=True,
            source="v4-v12",
        ),
    ]

    assert build_benchmark_assignment(evidence) == {"q1": "A", "q2": "B"}
