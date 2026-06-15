"""脚本 10：用当前答案解析器重算既有运行结果。

用于修复截断 JSON 等解析问题，不重新调用 Qwen。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.io.jsonl import write_jsonl
from agent.reasoning.answer_parser import parse_answer
from agent.reasoning.solver import _extract_confidence


def main() -> None:
    """读取 answer_results.jsonl，重算 answer/confidence 后写出。"""
    parser = argparse.ArgumentParser(description="Reparse saved answer_results without calling Qwen.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixed = [reparse_row(row) for row in rows]
    write_jsonl(args.output, fixed)
    changed = [
        (old["qid"], old.get("answer"), new.get("answer"), old.get("confidence"), new.get("confidence"))
        for old, new in zip(rows, fixed)
        if old.get("answer") != new.get("answer") or old.get("confidence") != new.get("confidence")
    ]
    print(f"wrote reparsed results -> {args.output}")
    for item in changed:
        print("changed", item)


def reparse_row(row: dict) -> dict:
    """重算单行结果；逐选项策略不改聚合答案，只修 raw_response 型单次回答。"""
    row = copy.deepcopy(row)
    answer_format = row.get("metadata", {}).get("answer_format", "")
    raw_response = row.get("raw_response", "")
    if raw_response and answer_format:
        answer = parse_answer(raw_response, answer_format)
        if answer:
            row["answer"] = answer
        row["confidence"] = _extract_confidence(raw_response)
    return row


if __name__ == "__main__":
    main()
