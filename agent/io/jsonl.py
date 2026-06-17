"""JSON/JSONL 文件读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


def read_jsonl(path: Path) -> Iterator[dict]:
    """逐行读取 JSONL；文件不存在时返回空迭代。"""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """写入 JSONL，保留中文字符。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    """追加单行 JSONL，用于长任务逐题 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def append_jsonl_rows(path: Path, rows: Iterable[dict]) -> int:
    """批量追加多行 JSONL，减少高频 open/close 开销。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        f.flush()
    return count


def read_json(path: Path) -> dict | list:
    """读取普通 JSON 文件。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict | list) -> None:
    """写入缩进后的 JSON，便于人工审计 evidence/token_usage。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
