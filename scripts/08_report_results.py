"""脚本 08：汇总一次运行结果的 Token、证据覆盖和格式风险。"""

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


def main() -> None:
    """读取 answer_results.jsonl 和可选 sample_manifest.json，生成诊断报告。"""
    parser = argparse.ArgumentParser(description="Report answer/token/evidence diagnostics.")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    results_path = args.results or settings.outputs_dir / "answer_results.jsonl"
    manifest_path = args.manifest or settings.outputs_dir / "sample_manifest.json"
    output_path = args.output or settings.outputs_dir / "run_report.md"

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = _load_manifest(manifest_path)
    report = build_report(rows, manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report["markdown"], encoding="utf-8")
    write_json(output_path.with_suffix(".json"), report["data"])
    print(f"wrote report -> {output_path}")


def build_report(rows: list[dict], manifest: dict[str, dict]) -> dict:
    """生成 Markdown 和结构化诊断数据。"""
    by_domain = defaultdict(lambda: _empty_stats())
    by_format = defaultdict(lambda: _empty_stats())
    issues = []
    detail_rows = []
    for row in rows:
        meta = manifest.get(row["qid"], {})
        domain = meta.get("domain") or row.get("metadata", {}).get("domain", "")
        answer_format = meta.get("answer_format") or row.get("metadata", {}).get("answer_format", "")
        usage = row["token_usage"]
        docs = sorted(set(item["doc_id"] for item in row.get("evidence", [])))
        strategy = row.get("metadata", {}).get("strategy", "single_pass")
        _add_stats(by_domain[domain], usage)
        _add_stats(by_format[answer_format], usage)
        issue = _diagnose_issue(row, meta, docs)
        if issue:
            issues.append(issue)
        detail_rows.append(
            {
                "qid": row["qid"],
                "domain": domain,
                "answer_format": answer_format,
                "answer": row["answer"],
                "confidence": row.get("confidence", 0.0),
                "strategy": strategy,
                "evidence_docs": docs,
                "token_usage": usage,
            }
        )

    total = {
        key: sum(row["token_usage"][key] for row in rows)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    data = {
        "total": total,
        "by_domain": dict(by_domain),
        "by_format": dict(by_format),
        "issues": issues,
        "details": detail_rows,
    }
    return {"markdown": _to_markdown(data), "data": data}


def _load_manifest(path: Path) -> dict[str, dict]:
    """读取抽样清单；没有清单时返回空字典。"""
    if not path.exists():
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {item["qid"]: item for item in items}


def _empty_stats() -> dict:
    """初始化聚合桶。"""
    return {"n": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _add_stats(bucket: dict, usage: dict) -> None:
    """累加 Token 统计。"""
    bucket["n"] += 1
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        bucket[key] += int(usage.get(key, 0))


def _diagnose_issue(row: dict, manifest: dict, docs: list[str]) -> dict | None:
    """检查答案格式、证据文档覆盖和逐选项空判断。"""
    answer = row.get("answer", "")
    answer_format = manifest.get("answer_format") or row.get("metadata", {}).get("answer_format", "")
    legal = bool(answer) and set(answer).issubset(set("ABCD"))
    if answer_format != "multi":
        legal = legal and len(answer) == 1
    else:
        legal = legal and answer == "".join(sorted(set(answer)))
    missing_docs = sorted(set(manifest.get("doc_ids") or []) - set(docs))
    none_opts = [
        item.get("option")
        for item in row.get("metadata", {}).get("option_judgements", [])
        if item.get("verdict") is None
    ]
    if legal and not missing_docs and not none_opts:
        return None
    return {
        "qid": row.get("qid"),
        "answer": answer,
        "answer_format": answer_format,
        "legal": legal,
        "missing_docs": missing_docs,
        "none_options": none_opts,
        "evidence_docs": docs,
    }


def _to_markdown(data: dict) -> str:
    """把诊断数据渲染为 Markdown。"""
    lines = [
        "# Run Report",
        "",
        "## Total",
        "",
        "| prompt_tokens | completion_tokens | total_tokens |",
        "|---:|---:|---:|",
        "| {prompt_tokens} | {completion_tokens} | {total_tokens} |".format(**data["total"]),
        "",
        "## By Domain",
        "",
        "| domain | n | prompt | completion | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in sorted(data["by_domain"].items()):
        lines.append(
            f"| {key} | {value['n']} | {value['prompt_tokens']} | "
            f"{value['completion_tokens']} | {value['total_tokens']} |"
        )
    lines.extend(["", "## By Format", "", "| format | n | prompt | completion | total |", "|---|---:|---:|---:|---:|"])
    for key, value in sorted(data["by_format"].items()):
        lines.append(
            f"| {key} | {value['n']} | {value['prompt_tokens']} | "
            f"{value['completion_tokens']} | {value['total_tokens']} |"
        )
    lines.extend(["", "## Issues", ""])
    if not data["issues"]:
        lines.append("No format/evidence/verdict issues detected.")
    else:
        for issue in data["issues"]:
            lines.append(f"- `{issue['qid']}`: {issue}")
    lines.extend(["", "## Details", "", "| qid | domain | format | answer | strategy | total_tokens | evidence_docs |"])
    lines.append("|---|---|---|---|---|---:|---|")
    for row in data["details"]:
        docs = ",".join(row["evidence_docs"])
        lines.append(
            f"| {row['qid']} | {row['domain']} | {row['answer_format']} | {row['answer']} | "
            f"{row['strategy']} | {row['token_usage']['total_tokens']} | {docs} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
