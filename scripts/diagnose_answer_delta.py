"""Answer delta diagnostics for A-board runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.io.jsonl import write_json
from agent.io.output_layout import resolve_compare_dir



def load_results(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["qid"]: row for row in rows}



def compare_answer_results(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = load_results(baseline_path)
    candidate = load_results(candidate_path)
    common_qids = sorted(set(baseline) & set(candidate))
    by_domain = defaultdict(int)
    by_format = defaultdict(int)
    rows = []
    changed_rows = []
    token_delta_total = 0
    evidence_doc_changes = 0

    for qid in common_qids:
        base = baseline[qid]
        cand = candidate[qid]
        base_meta = base.get("metadata", {})
        cand_meta = cand.get("metadata", {})
        domain = cand_meta.get("domain") or base_meta.get("domain", "")
        answer_format = cand_meta.get("answer_format") or base_meta.get("answer_format", "")
        base_docs = _docs(base)
        cand_docs = _docs(cand)
        token_delta = int(cand.get("token_usage", {}).get("total_tokens", 0)) - int(base.get("token_usage", {}).get("total_tokens", 0))
        row = {
            "qid": qid,
            "domain": domain,
            "answer_format": answer_format,
            "baseline_answer": base.get("answer", ""),
            "candidate_answer": cand.get("answer", ""),
            "answer_changed": base.get("answer", "") != cand.get("answer", ""),
            "token_delta": token_delta,
            "baseline_docs": base_docs,
            "candidate_docs": cand_docs,
        }
        rows.append(row)
        by_domain[domain] += 1
        by_format[answer_format] += 1
        token_delta_total += token_delta
        if base_docs != cand_docs:
            evidence_doc_changes += 1
        if row["answer_changed"]:
            changed_rows.append(row)

    data = {
        "question_count": len(common_qids),
        "answer_change_count": len(changed_rows),
        "token_delta_total": token_delta_total,
        "evidence_doc_changes": evidence_doc_changes,
        "by_domain": dict(by_domain),
        "by_format": dict(by_format),
        "rows": rows,
        "changed_rows": changed_rows,
    }
    return {"markdown": render_markdown(data), "data": data}



def render_markdown(data: dict) -> str:
    lines = [
        "# Answer Delta Diagnostics",
        "",
        f"Questions: {data['question_count']}",
        f"Answer changes: {data['answer_change_count']}",
        f"Evidence doc changes: {data['evidence_doc_changes']}",
        "",
        "## Token Delta",
        "",
        f"Total token delta: {data['token_delta_total']}",
        "",
        "## By Domain",
        "",
        "| domain | count |",
        "|---|---:|",
    ]
    for key, count in sorted(data["by_domain"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## By Format", "", "| format | count |", "|---|---:|"])
    for key, count in sorted(data["by_format"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Answer Changes", ""])
    if not data["changed_rows"]:
        lines.append("No answer changes.")
    else:
        for row in data["changed_rows"]:
            lines.append(
                f"- `{row['qid']}` ({row['domain']}/{row['answer_format']}): "
                f"{row['baseline_answer']} -> {row['candidate_answer']} (token delta {row['token_delta']})"
            )
    lines.extend(["", "## Details", "", "| qid | domain | format | baseline | candidate | token_delta | baseline_docs | candidate_docs |"])
    lines.append("|---|---|---|---|---|---:|---|---|")
    for row in data["rows"]:
        lines.append(
            f"| {row['qid']} | {row['domain']} | {row['answer_format']} | {row['baseline_answer']} | "
            f"{row['candidate_answer']} | {row['token_delta']} | {','.join(row['baseline_docs'])} | {','.join(row['candidate_docs'])} |"
        )
    return "\n".join(lines) + "\n"



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answer delta diagnostics.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    report = compare_answer_results(args.baseline, args.candidate)
    output_path = args.output or resolve_compare_dir(
        settings,
        scope="a100" if report["data"]["question_count"] == 100 else "sample",
        baseline_slug=args.baseline.parent.name,
        candidate_slug=args.candidate.parent.name,
    ) / "answer_delta.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report["markdown"], encoding="utf-8")
    write_json(output_path.with_suffix(".json"), report["data"])
    print(f"wrote answer delta -> {output_path}")



def _docs(row: dict) -> list[str]:
    return sorted(set(item["doc_id"] for item in row.get("evidence", [])))


if __name__ == "__main__":
    main()
