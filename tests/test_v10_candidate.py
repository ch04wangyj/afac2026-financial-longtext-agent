import json
from pathlib import Path

from agent.evaluation.residual_profiles import resolve_profile_reviews


ROOT = Path(__file__).resolve().parents[1]


def test_v10_inferred_profile_contains_only_four_pattern_gains():
    payload = json.loads(
        (ROOT / "configs" / "v10_inferred_reviews.json").read_text(encoding="utf-8")
    )

    reviews = resolve_profile_reviews(payload["profiles"], "inferred90")

    assert {qid: row["answer"] for qid, row in reviews.items()} == {
        "fc_a_014": "AB",
        "reg_a_004": "AC",
        "res_a_003": "A",
        "res_a_011": "ABC",
    }
