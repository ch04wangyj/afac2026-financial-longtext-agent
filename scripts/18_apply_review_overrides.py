"""脚本 18：应用 V13 人工证据复核覆盖，不改变 Token 统计。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.evaluation.review_overrides import apply_review_overrides
from agent.io.jsonl import read_jsonl, write_jsonl
from agent.schemas import AnswerResult


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed answer overrides to V13 results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, default=ROOT / "configs" / "v13_review_overrides.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [AnswerResult.from_dict(row) for row in read_jsonl(args.results)]
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    reviewed = apply_review_overrides(rows, overrides)
    write_jsonl(args.output, (row.to_dict() for row in reviewed))
    changed = [row.qid for row in reviewed if row.metadata.get("final_review")]
    print(f"wrote {len(reviewed)} reviewed results ({len(changed)} audited overrides) -> {args.output}")


if __name__ == "__main__":
    main()
