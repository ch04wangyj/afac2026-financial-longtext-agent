"""重型 Python 任务守护，避免重复长任务叠加占用内存。"""

from __future__ import annotations

import os

import psutil


def find_heavy_python_jobs(current_pid: int | None = None, min_rss_gb: float = 1.0) -> list[dict[str, object]]:
    current_pid = os.getpid() if current_pid is None else current_pid
    rows: list[dict[str, object]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        pid = proc.info.get("pid")
        if pid == current_pid:
            continue
        name = str(proc.info.get("name") or "").lower()
        if "python" not in name:
            continue
        mem = proc.info.get("memory_info")
        rss_gb = (mem.rss / 1024**3) if mem else 0.0
        if rss_gb < min_rss_gb:
            continue
        rows.append(
            {
                "pid": pid,
                "rss_gb": round(rss_gb, 2),
                "cmdline": " ".join(proc.info.get("cmdline") or []),
            }
        )
    return sorted(rows, key=lambda row: row["rss_gb"], reverse=True)


def assert_no_other_heavy_python_jobs(min_rss_gb: float = 1.0) -> None:
    conflicts = find_heavy_python_jobs(min_rss_gb=min_rss_gb)
    if conflicts:
        raise RuntimeError(f"Heavy Python job already running: {conflicts}")
