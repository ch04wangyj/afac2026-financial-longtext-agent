"""脚本 09：比较两次运行的答案和 Token 变化。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.io.jsonl import write_json
from agent.io.output_layout import resolve_compare_dir


def main() -> None:
    """比较 baseline 与 candidate 的 answer_results.jsonl。"""
    parser = argparse.ArgumentParser(description="Compare two answer_results.jsonl runs.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    report = compare_runs(baseline, candidate)
    output_path = args.output or _default_output_path(
        settings,
        args.baseline,
        args.candidate,
        report["data"]["question_count"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report["markdown"], encoding="utf-8")
    write_json(output_path.with_suffix(".json"), report["data"])
    print(f"wrote comparison -> {output_path}")


def compare_runs(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    """生成运行对比报告。"""
    common_qids = sorted(set(baseline) & set(candidate))
    rows = []
    answer_changes = []
    total_base = _empty_usage()
    total_cand = _empty_usage()
    for qid in common_qids:
        base = baseline[qid]
        cand = candidate[qid]
        base_usage = base["token_usage"]
        cand_usage = cand["token_usage"]
        _add_usage(total_base, base_usage)
        _add_usage(total_cand, cand_usage)
        row = {
            "qid": qid,
            "baseline_answer": base.get("answer", ""),
            "candidate_answer": cand.get("answer", ""),
            "answer_changed": base.get("answer", "") != cand.get("answer", ""),
            "baseline_total_tokens": base_usage.get("total_tokens", 0),
            "candidate_total_tokens": cand_usage.get("total_tokens", 0),
            "token_delta": cand_usage.get("total_tokens", 0) - base_usage.get("total_tokens", 0),
            "baseline_docs": _docs(base),
            "candidate_docs": _docs(cand),
        }
        if row["answer_changed"]:
            answer_changes.append(row)
        rows.append(row)
    total_delta = {
        key: total_cand[key] - total_base[key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    data = {
        "question_count": len(common_qids),
        "answer_change_count": len(answer_changes),
        "baseline_total": total_base,
        "candidate_total": total_cand,
        "delta": total_delta,
        "rows": rows,
        "answer_changes": answer_changes,
    }
    return {"markdown": _to_markdown(data), "data": data}


def _load(path: Path) -> dict[str, dict]:
    """按 qid 加载运行结果。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["qid"]: row for row in rows}



def _default_output_path(settings: Settings, baseline: Path, candidate: Path, question_count: int) -> Path:
    scope = "a100" if question_count == 100 else "sample"
    out_dir = resolve_compare_dir(
        settings,
        scope=scope,
        baseline_slug=baseline.parent.name,
        candidate_slug=candidate.parent.name,
    )
    return out_dir / "comparison.md"



def _empty_usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_usage(total: dict[str, int], usage: dict) -> None:
    """累加 Token。"""
    for key in total:
        total[key] += int(usage.get(key, 0))


def _docs(row: dict) -> list[str]:
    """提取本题最终 evidence 覆盖的 doc_id。"""
    return sorted(set(item["doc_id"] for item in row.get("evidence", [])))


def _to_markdown(data: dict) -> str:
    """渲染 Markdown 对比报告。"""
    lines = [
        "# Run Comparison",
        "",
        f"Questions: {data['question_count']}",
        f"Answer changes: {data['answer_change_count']}",
        "",
        "## Token Delta",
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        lines.append(
            f"| {key} | {data['baseline_total'][key]} | {data['candidate_total'][key]} | {data['delta'][key]} |"
        )
    lines.extend(["", "## Answer Changes", ""])
    if not data["answer_changes"]:
        lines.append("No answer changes.")
    else:
        for row in data["answer_changes"]:
            lines.append(
                f"- `{row['qid']}`: {row['baseline_answer']} -> {row['candidate_answer']} "
                f"(delta {row['token_delta']})"
            )
    lines.extend(["", "## Details", "", "| qid | baseline | candidate | delta | baseline_docs | candidate_docs |"])
    lines.append("|---|---|---|---:|---|---|")
    for row in data["rows"]:
        lines.append(
            f"| {row['qid']} | {row['baseline_answer']} | {row['candidate_answer']} | "
            f"{row['token_delta']} | {','.join(row['baseline_docs'])} | {','.join(row['candidate_docs'])} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
