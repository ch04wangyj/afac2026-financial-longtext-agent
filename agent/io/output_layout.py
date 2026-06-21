"""outputs 目录结构与默认落盘路径辅助。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


_SCOPE_ROOTS = {
    "test": "tests",
    "sample": "samples",
    "a100": "a100",
}


def slugify(value: str) -> str:
    """将策略名/目录名规整为安全 slug。"""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "default"


def timestamp_slug(stamp: str | None = None) -> str:
    """统一时间戳格式。"""
    return stamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")


def infer_artifact_dir_from_results(results_path: Path) -> Path:
    """后处理产物默认写回 answer_results.jsonl 所在目录。"""
    return results_path.resolve().parent


def resolve_run_dir(
    settings,
    *,
    run_scope: str,
    run_name: str,
    strategy: str,
    dry_run: bool,
    stamp: str | None = None,
) -> Path:
    """按 run 类型生成标准输出目录。"""
    root = Path(settings.outputs_dir)
    scope_root = _SCOPE_ROOTS[run_scope]
    mode = "dry" if dry_run else "live"
    return root / scope_root / slugify(run_name) / mode / f"{timestamp_slug(stamp)}_{slugify(strategy)}"


def resolve_compare_dir(settings, *, scope: str, baseline_slug: str, candidate_slug: str) -> Path:
    """生成对比报告目录。"""
    root = Path(settings.outputs_dir)
    scope_root = _SCOPE_ROOTS[scope]
    return root / scope_root / "compare" / f"{slugify(baseline_slug)}__vs__{slugify(candidate_slug)}"


def choose_output_dir(
    settings,
    *,
    run_scope: str,
    run_name: str,
    strategy: str,
    dry_run: bool,
    stamp: str | None = None,
    explicit_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    """若用户显式指定输出目录则尊重，否则按规范目录自动生成。"""
    if explicit_dir is not None:
        return Path(explicit_dir).expanduser().resolve()
    if getattr(settings, "has_explicit_outputs_dir", False):
        return Path(settings.outputs_dir)
    if resume:
        latest = _latest_matching_run_dir(
            settings,
            run_scope=run_scope,
            run_name=run_name,
            strategy=strategy,
            dry_run=dry_run,
        )
        if latest is not None:
            return latest
    return resolve_run_dir(
        settings,
        run_scope=run_scope,
        run_name=run_name,
        strategy=strategy,
        dry_run=dry_run,
        stamp=stamp,
    )


def _latest_matching_run_dir(settings, *, run_scope: str, run_name: str, strategy: str, dry_run: bool) -> Path | None:
    root = Path(settings.outputs_dir)
    scope_root = _SCOPE_ROOTS[run_scope]
    mode = "dry" if dry_run else "live"
    base = root / scope_root / slugify(run_name) / mode
    if not base.exists():
        return None
    suffix = f"_{slugify(strategy)}"
    matches = [path for path in base.iterdir() if path.is_dir() and path.name.endswith(suffix)]
    if not matches:
        return None
    return sorted(matches)[-1]
