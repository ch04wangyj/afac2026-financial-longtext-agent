import hashlib
import json
from pathlib import Path

import pytest

from agent.evaluation.leaderboard_registry import (
    load_run_registry,
    load_verified_leaderboard_runs,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_result(path, answers):
    rows = [
        {
            "qid": qid,
            "answer": answer,
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        for qid, answer in answers.items()
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_registry_excludes_unverified_runs_and_checks_hash(tmp_path):
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "outputs"
    config_dir.mkdir()
    output_dir.mkdir()
    answers = {f"q{index:03d}": "A" for index in range(100)}
    verified_paths = []
    for name in ("a", "b"):
        result_path = output_dir / f"{name}.jsonl"
        _write_result(result_path, answers)
        verified_paths.append(result_path)
    legacy_path = output_dir / "legacy.jsonl"
    _write_result(legacy_path, answers)

    runs = []
    for name, result_path in zip(("a", "b"), verified_paths):
        runs.append(
            {
                "name": name,
                "status": "verified_submission",
                "usable_for_constraints": True,
                "result_path": f"outputs/{result_path.name}",
                "correct_count": 100,
                "total_tokens": 200,
                "official_score": 100.0,
                "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        )
    runs.append(
        {
            "name": "legacy",
            "status": "legacy_unverified",
            "usable_for_constraints": False,
            "result_path": "outputs/legacy.jsonl",
            "correct_count": 100,
            "total_tokens": 200,
            "official_score": 100.0,
            "sha256": hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
        }
    )
    registry = config_dir / "runs.json"
    registry.write_text(
        json.dumps({"schema_version": 1, "runs": runs}),
        encoding="utf-8",
    )

    assert len(load_run_registry(registry)) == 3
    assert [run.name for run in load_verified_leaderboard_runs(registry)] == [
        "a",
        "b",
    ]

    verified_paths[0].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="哈希不匹配"):
        load_verified_leaderboard_runs(registry)


def test_project_registry_uses_reproducible_compact_snapshots():
    runs = load_verified_leaderboard_runs(
        ROOT / "configs" / "leaderboard_runs.json"
    )

    assert [run.name for run in runs] == [
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
    ]
    assert runs[-1].correct_count == 85
    assert runs[-1].answers["reg_a_004"] == "A"
