from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_cleanup_module():
    path = ROOT / "scripts" / "10_cleanup_outputs.py"
    spec = importlib.util.spec_from_file_location("cleanup_outputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def test_cleanup_plan_keeps_whitelisted_runs_and_marks_probe_dirs(tmp_path: Path):
    module = _load_cleanup_module()
    outputs = tmp_path / "outputs"
    keep_dir = outputs / "a100" / "live" / "2026-06-18_090743_logicrag_agent"
    stale_dir = outputs / "qwen_plus_probe_fc_a_003"
    root_file = outputs / "answer.csv"

    keep_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    outputs.mkdir(exist_ok=True)
    root_file.write_text("qid,answer\n", encoding="utf-8")

    plan = module.build_cleanup_plan(outputs, keep_names={keep_dir.name})

    assert stale_dir in plan.delete_dirs
    assert root_file in plan.delete_files
    assert keep_dir not in plan.delete_dirs
