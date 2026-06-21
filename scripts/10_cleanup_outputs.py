"""脚本 10：安全清理 outputs 下过往无用数据。"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT_LEVEL_FILES = {
    "answer.csv",
    "answer_results.jsonl",
    "evidence.json",
    "token_usage.json",
    "run_report.md",
    "run_report.json",
}
STALE_DIR_PATTERNS = (
    r"^qwen_plus_.*",
    r"^smoke_insurance_.*",
    r"^compare_.*",
)


@dataclass(frozen=True)
class CleanupPlan:
    delete_dirs: list[Path]
    delete_files: list[Path]



def build_cleanup_plan(outputs_dir: Path, keep_names: set[str] | None = None) -> CleanupPlan:
    """仅根据目录名/文件名生成清理计划，不做实际删除。"""
    keep = keep_names or set()
    delete_dirs: list[Path] = []
    delete_files: list[Path] = []
    if not outputs_dir.exists():
        return CleanupPlan(delete_dirs=delete_dirs, delete_files=delete_files)

    for path in outputs_dir.iterdir():
        if path.name in keep:
            continue
        if path.is_file() and path.name in ROOT_LEVEL_FILES:
            delete_files.append(path)
            continue
        if path.is_dir() and any(re.match(pattern, path.name) for pattern in STALE_DIR_PATTERNS):
            delete_dirs.append(path)

    return CleanupPlan(
        delete_dirs=sorted(delete_dirs, key=lambda item: item.name),
        delete_files=sorted(delete_files, key=lambda item: item.name),
    )



def main() -> None:
    """默认 dry-run；只有 --apply 时才真正删除。"""
    parser = argparse.ArgumentParser(description="Safely clean stale data under outputs/.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--keep-dir", action="append", default=[], help="Directory name to keep even if it matches cleanup rules.")
    parser.add_argument("--dry-run", action="store_true", help="Preview matches without deleting anything.")
    parser.add_argument("--apply", action="store_true", help="Actually delete the matched files/directories.")
    args = parser.parse_args()

    plan = build_cleanup_plan(args.outputs_dir, keep_names=set(args.keep_dir))

    mode = "apply" if args.apply else "dry-run"
    print(f"outputs_dir={args.outputs_dir}")
    print(f"mode={mode}")
    print("delete_files:")
    for path in plan.delete_files:
        print(f"  FILE {path}")
    print("delete_dirs:")
    for path in plan.delete_dirs:
        print(f"  DIR  {path}")

    if not args.apply:
        return

    for path in plan.delete_files:
        path.unlink(missing_ok=True)
    for path in plan.delete_dirs:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
