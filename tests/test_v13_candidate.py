import csv
import json
from pathlib import Path

from agent.evaluation.residual_profiles import resolve_profile_reviews


ROOT = Path(__file__).resolve().parents[1]


def test_v13_safe_gain_profile_only_contains_no_downside_changes():
    payload = json.loads(
        (ROOT / "configs" / "v13_safe_gain_reviews.json").read_text(
            encoding="utf-8"
        )
    )

    reviews = resolve_profile_reviews(payload["profiles"], "safe_gain")

    assert {qid: row["answer"] for qid, row in reviews.items()} == {
        "reg_a_004": "ABC",
        "res_a_011": "ABCD",
    }
    assert all(
        row["decision"] == "leaderboard_no_downside_candidate"
        for row in reviews.values()
    )


def test_v13_submission_mirror_has_complete_rows_and_token_conservation():
    with (ROOT / "submissions" / "v13_answer.csv").open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 101
    assert rows[0]["qid"] == "summary"
    answers = rows[1:]
    assert len({row["qid"] for row in answers}) == 100
    assert sum(int(row["prompt_tokens"]) for row in answers) == int(
        rows[0]["prompt_tokens"]
    )
    assert sum(int(row["completion_tokens"]) for row in answers) == int(
        rows[0]["completion_tokens"]
    )
    assert sum(int(row["total_tokens"]) for row in answers) == int(
        rows[0]["total_tokens"]
    )
