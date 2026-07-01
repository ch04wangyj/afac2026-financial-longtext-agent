from agent.evaluation.active_probe import evaluate_probe, rank_probe_variants
from agent.evaluation.leaderboard_constraints import LeaderboardRun


def test_probe_enumerates_non_contiguous_possible_scores():
    runs = [
        LeaderboardRun(
            name="aa",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="bb",
            answers={"q1": "B", "q2": "B"},
            correct_count=1,
        ),
    ]
    valid = {"q1": {"A", "B"}, "q2": {"A", "B"}}

    result = evaluate_probe(
        runs,
        candidate_answers={"q1": "B", "q2": "A"},
        reference_answers=runs[0].answers,
        reference_correct_count=1,
        valid_answers_by_qid=valid,
    )

    assert result.possible_correct_counts == (0, 2)
    assert result.min_delta == -1
    assert result.max_delta == 1
    assert result.outcome_information_bits == 1.0


def test_probe_ranking_prioritizes_worst_case_score():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="known",
            answers={"q1": "B", "q2": "A"},
            correct_count=2,
        ),
    ]
    valid = {"q1": {"A", "B"}, "q2": {"A", "B"}}

    ranked = rank_probe_variants(
        runs,
        reference_answers=runs[0].answers,
        reference_correct_count=1,
        alternatives_by_qid={"q1": ("A", "B"), "q2": ("A", "B")},
        valid_answers_by_qid=valid,
    )

    assert ranked[0].min_correct == 2
    assert ranked[0].changes == (("q1", "A", "B"),)
