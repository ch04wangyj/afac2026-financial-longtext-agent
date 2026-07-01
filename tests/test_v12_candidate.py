import json
from pathlib import Path

from agent.evaluation.residual_profiles import resolve_profile_reviews


ROOT = Path(__file__).resolve().parents[1]


def test_v12_rejected_profile_is_retained_for_reproduction():
    payload = json.loads(
        (ROOT / "configs" / "v12_conditioned_reviews.json").read_text(
            encoding="utf-8"
        )
    )

    reviews = resolve_profile_reviews(payload["profiles"], "target93")

    assert payload["profiles"]["target93"]["description"].startswith("历史负向实验")
    assert {qid: row["answer"] for qid, row in reviews.items()} == {
        "fc_a_014": "A",
        "reg_a_001": "C",
        "reg_a_004": "A",
        "res_a_011": "B",
        "res_a_017": "AB",
        "res_a_018": "B",
    }
    assert reviews["res_a_017"]["confidence"] == "verified"
