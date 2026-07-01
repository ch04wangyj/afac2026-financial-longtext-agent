from types import SimpleNamespace

import numpy as np

from agent.evaluation.leaderboard_constraints import (
    LeaderboardRun,
    _build_model,
    infer_correctness_bounds,
    infer_question_constraints,
    infer_weighted_assignment,
    is_partial_assignment_feasible,
)


def test_leaderboard_constraints_find_forced_regression():
    runs = [
        LeaderboardRun(
            name="baseline",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "A"},
            correct_count=2,
        ),
    ]

    results = infer_question_constraints(
        runs,
        baseline_name="baseline",
        valid_answers_by_qid={"q1": {"A", "B"}, "q2": {"A", "B"}},
    )
    by_qid = {row.qid: row for row in results}

    assert by_qid["q1"].baseline_forced_wrong is True
    assert by_qid["q1"].forced_observed_answer == "B"
    assert by_qid["q2"].baseline_forced_correct is True


def test_leaderboard_constraints_reject_inconsistent_counts():
    runs = [
        LeaderboardRun(name="a", answers={"q1": "A"}, correct_count=1),
        LeaderboardRun(name="b", answers={"q1": "A"}, correct_count=0),
    ]

    try:
        infer_question_constraints(runs, baseline_name="a")
    except RuntimeError as exc:
        assert "不存在可行解" in str(exc)
    else:
        raise AssertionError("不一致的官网正确题数必须被拒绝")


def test_weighted_assignment_respects_official_counts_before_votes():
    runs = [
        LeaderboardRun(
            name="baseline",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="official_candidate",
            answers={"q1": "B", "q2": "A"},
            correct_count=2,
        ),
    ]
    assignment = infer_weighted_assignment(
        runs,
        baseline_name="baseline",
        valid_answers_by_qid={"q1": {"A", "B"}, "q2": {"A", "B"}},
        answer_weights={
            "q1": {"A": 100.0, "B": 1.0},
            "q2": {"B": 100.0, "A": 1.0},
        },
    )

    assert assignment == {"q1": "B", "q2": "A"}


def test_partial_assignment_checks_joint_feasibility():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "B"},
            correct_count=1,
        ),
    ]
    valid = {"q1": {"A", "B"}, "q2": {"A", "B"}}

    assert is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid=valid,
        partial_assignment={"q1": "A", "q2": "B"},
    )
    assert not is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid=valid,
        partial_assignment={"q1": "A", "q2": "A"},
    )


def test_forbidden_answers_can_express_old_and_new_both_wrong():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "A"},
            correct_count=1,
        ),
    ]
    valid = {"q1": {"A", "B", "C"}, "q2": {"A", "B", "C"}}

    assert is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid=valid,
        partial_assignment={},
        forbidden_answers_by_qid={"q1": {"A", "B"}},
    )
    assert not is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid={"q1": {"A", "B"}, "q2": {"A", "B"}},
        partial_assignment={},
        forbidden_answers_by_qid={"q1": {"A", "B"}},
    )
    assert not is_partial_assignment_feasible(
        runs,
        valid_answers_by_qid=valid,
        partial_assignment={"q1": "A"},
        forbidden_answers_by_qid={"q1": {"A"}},
    )


def test_conditioned_constraints_protect_additional_baseline_answer():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "B"},
            correct_count=1,
        ),
    ]
    valid = {"q1": {"A", "B"}, "q2": {"A", "B"}}

    results = infer_question_constraints(
        runs,
        baseline_name="base",
        valid_answers_by_qid=valid,
        partial_assignment={"q1": "A"},
    )
    by_qid = {row.qid: row for row in results}

    assert by_qid["q1"].baseline_forced_correct is True
    assert by_qid["q2"].baseline_forced_wrong is True
    assert by_qid["q2"].forced_observed_answer == "B"


def test_conditioned_weighted_assignment_never_overrides_fixed_answer():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A"},
            correct_count=1,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "B"},
            correct_count=1,
        ),
    ]
    assignment = infer_weighted_assignment(
        runs,
        baseline_name="base",
        valid_answers_by_qid={"q1": {"A", "B"}, "q2": {"A", "B"}},
        answer_weights={"q1": {"B": 100.0}, "q2": {"A": 100.0}},
        partial_assignment={"q1": "A"},
    )

    assert assignment == {"q1": "A", "q2": "B"}


def test_correctness_bounds_support_question_subsets():
    runs = [
        LeaderboardRun(
            name="base",
            answers={"q1": "A", "q2": "A", "q3": "A"},
            correct_count=2,
        ),
        LeaderboardRun(
            name="candidate",
            answers={"q1": "B", "q2": "A", "q3": "A"},
            correct_count=3,
        ),
    ]
    valid = {qid: {"A", "B"} for qid in ("q1", "q2", "q3")}

    unconstrained = infer_correctness_bounds(
        runs,
        baseline_answers=runs[0].answers,
        valid_answers_by_qid=valid,
        subset_qids={"q2", "q3"},
    )
    conditioned = infer_correctness_bounds(
        runs,
        baseline_answers=runs[0].answers,
        valid_answers_by_qid=valid,
        partial_assignment={"q2": "A"},
        subset_qids={"q2", "q3"},
    )

    assert unconstrained.to_dict() == {
        "question_count": 2,
        "min_correct": 2,
        "max_correct": 2,
        "min_wrong": 0,
        "max_wrong": 0,
    }
    assert conditioned.min_correct == 2
    assert conditioned.max_correct == 2


def test_milp_rejects_presolve_solution_that_violates_original_constraints(
    monkeypatch,
):
    """求解器不能把 HiGHS presolve 的伪可行解当成排行榜结论。"""
    runs = [
        LeaderboardRun(name="base", answers={"q1": "A"}, correct_count=1),
    ]
    model = _build_model(
        runs,
        ["q1"],
        {"q1": ("A", "B")},
    )
    calls: list[bool] = []

    def fake_milp(*, options, **_kwargs):
        calls.append(options["presolve"])
        if options["presolve"]:
            # q1=B 违反 base 必须命中 1 题的等式约束。
            return SimpleNamespace(success=True, x=np.array([0.0, 1.0]))
        return SimpleNamespace(success=False, x=None)

    monkeypatch.setattr("scipy.optimize.milp", fake_milp)

    answer_b = model.variable_by_state[("q1", "B")]
    assert model.optimize(forced_variable=answer_b) is None
    assert calls == [True, False]
