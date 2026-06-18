from __future__ import annotations

from pathlib import Path

from agent.config import Settings
from agent.io.output_layout import choose_output_dir, infer_artifact_dir_from_results, resolve_compare_dir, resolve_run_dir


def test_resolve_smoke_live_dir_under_tests_bucket(tmp_path: Path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")

    path = resolve_run_dir(
        settings,
        run_scope="test",
        run_name="smoke",
        strategy="logicrag_agent",
        dry_run=False,
        stamp="2026-06-18_100727",
    )

    assert path == tmp_path / "outputs" / "tests" / "smoke" / "live" / "2026-06-18_100727_logicrag_agent"



def test_resolve_sample20_dir_under_samples_bucket(tmp_path: Path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")

    path = resolve_run_dir(
        settings,
        run_scope="sample",
        run_name="sample20",
        strategy="logicrag_agent",
        dry_run=True,
        stamp="2026-06-18_100727",
    )

    assert path == tmp_path / "outputs" / "samples" / "sample20" / "dry" / "2026-06-18_100727_logicrag_agent"



def test_resolve_a100_compare_dir(tmp_path: Path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")

    path = resolve_compare_dir(settings, scope="a100", baseline_slug="prev_full", candidate_slug="parallel_thinking")

    assert path == tmp_path / "outputs" / "a100" / "compare" / "prev_full__vs__parallel_thinking"



def test_make_submission_defaults_to_results_parent(tmp_path: Path):
    results = tmp_path / "outputs" / "samples" / "sample20" / "live" / "2026-06-18_100727_logicrag" / "answer_results.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text('{"qid":"q1"}\n', encoding="utf-8")

    artifact_dir = infer_artifact_dir_from_results(results)

    assert artifact_dir == results.parent



def test_resolve_full_run_defaults_to_a100_bucket(tmp_path: Path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")

    path = resolve_run_dir(
        settings,
        run_scope="a100",
        run_name="full100",
        strategy="hybrid",
        dry_run=False,
        stamp="2026-06-18_100727",
    )

    assert path == tmp_path / "outputs" / "a100" / "full100" / "live" / "2026-06-18_100727_hybrid"



def test_choose_output_dir_uses_latest_matching_run_when_resuming(tmp_path: Path):
    settings = Settings(project_root=tmp_path, outputs_dir=tmp_path / "outputs")
    older = tmp_path / "outputs" / "samples" / "sample20" / "live" / "2026-06-17_235959_logicrag_agent"
    newer = tmp_path / "outputs" / "samples" / "sample20" / "live" / "2026-06-18_100727_logicrag_agent"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    path = choose_output_dir(
        settings,
        run_scope="sample",
        run_name="sample20",
        strategy="logicrag_agent",
        dry_run=False,
        resume=True,
    )

    assert path == newer
