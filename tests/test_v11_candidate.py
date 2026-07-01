import json
from pathlib import Path

from agent.evaluation.residual_profiles import resolve_profile_reviews


ROOT = Path(__file__).resolve().parents[1]


def test_v11_target90_profile_contains_four_ternary_reviews():
    payload = json.loads(
        (ROOT / "configs" / "v11_ternary_reviews.json").read_text(encoding="utf-8")
    )

    reviews = resolve_profile_reviews(payload["profiles"], "target90")

    assert {qid: row["answer"] for qid, row in reviews.items()} == {
        "reg_a_004": "ABC",
        "res_a_011": "ABCD",
        "res_a_017": "ABD",
        "res_a_018": "B",
    }
    assert reviews["res_a_011"]["confidence"] == "medium"
