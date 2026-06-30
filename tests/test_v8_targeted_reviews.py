import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v8_targeted_review_groups_are_disjoint():
    payload = json.loads(
        (ROOT / "configs" / "v8_targeted_reviews.json").read_text(encoding="utf-8")
    )
    seen: set[str] = set()

    for group in payload["groups"].values():
        assert group["candidate_result"].endswith("answer_results.jsonl")
        qids = set(group["reviews"])
        assert qids
        assert not (seen & qids)
        seen.update(qids)
