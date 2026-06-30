import pytest

from agent.evaluation.residual_profiles import resolve_profile_reviews


def test_v9_profile_inherits_core_reviews():
    profiles = {
        "core": {"reviews": {"q1": {"answer": "A"}}},
        "target90": {
            "extends": "core",
            "reviews": {"q2": {"answer": "B"}},
        },
    }

    reviews = resolve_profile_reviews(profiles, "target90")

    assert reviews == {
        "q1": {"answer": "A"},
        "q2": {"answer": "B"},
    }


def test_v9_profile_rejects_inheritance_cycle():
    profiles = {
        "a": {"extends": "b"},
        "b": {"extends": "a"},
    }

    with pytest.raises(ValueError, match="形成循环"):
        resolve_profile_reviews(profiles, "a")
