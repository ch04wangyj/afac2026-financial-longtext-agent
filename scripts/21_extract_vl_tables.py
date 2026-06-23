"""脚本 21：批量离线提取 B 类图像页的表格文本。

该脚本在文档预处理阶段调用 Qwen-VL，把图像化表格页转成结构化文本。
产生的 Token 不计入最终评测 Token，仅用于离线审计。

用法：
    python scripts/21_extract_vl_tables.py --scan-only
    python scripts/21_extract_vl_tables.py --output-dir processed_data/v15_vl_tables
    python scripts/21_extract_vl_tables.py --output-dir processed_data/v15_vl_tables --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.io.jsonl import append_jsonl, read_jsonl, write_jsonl
from agent.preprocess.vl_table_extract import (
    VLExtractConfig,
    VLTableResult,
    extract_table_from_page,
    scan_b_class_pages,
)
from agent.schemas import TokenUsage


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tables from B-class image pages using Qwen-VL (offline).")
    parser.add_argument("--scan-only", action="store_true", help="只扫描 B 类页，不调用 Qwen-VL。")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少页（0=全部）。")
    parser.add_argument("--workers", type=int, default=4, help="Qwen-VL 并发数。")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--vl-model", default="qwen-vl-max")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--resume", action="store_true", help="跳过已完成的页。")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
    output_dir = args.output_dir or settings.processed_dir / "v15_vl_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url or settings.qwen_base_url
    config = VLExtractConfig(dpi=args.dpi, vl_model=args.vl_model)

    scan_path = output_dir / "b_class_pages.jsonl"
    results_path = output_dir / "vl_table_results.jsonl"
    summary_path = output_dir / "vl_extraction_summary.json"

    print("[1/4] Scanning B-class image pages...")
    b_pages = scan_b_class_pages(settings.raw_root, config)
    write_jsonl(scan_path, [p.__dict__ for p in b_pages])
    print(f"  Found {len(b_pages)} B-class pages -> {scan_path}")
    by_domain = Counter(p.domain for p in b_pages)
    print(f"  By domain: {dict(by_domain)}")

    if args.scan_only:
        print("  --scan-only mode, skipping extraction.")
        return

    done_keys: set[str] = set()
    if args.resume and results_path.exists():
        for row in read_jsonl(results_path):
            done_keys.add(f"{row['doc_id']}:{row['page_index']}")
        print(f"  Resume: skipping {len(done_keys)} already-extracted pages.")

    pending = [p for p in b_pages if f"{p.doc_id}:{p.page_index}" not in done_keys]
    if args.limit:
        pending = pending[: args.limit]
    print(f"[2/4] Extracting tables from {len(pending)} pages with {args.workers} workers...")

    if not pending:
        print("  No pending pages.")
        _write_summary(summary_path, b_pages, [], TokenUsage())
        return

    total_usage = TokenUsage()
    results: list[dict] = []
    completed = 0
    failed = 0

    def _extract(page):
        pdf_path = settings.raw_root / page.domain / page.file_name
        try:
            result = extract_table_from_page(
                pdf_path,
                page.page_index,
                page.domain,
                page.doc_id,
                config,
                base_url,
            )
            return result
        except Exception as exc:
            return VLTableResult(
                domain=page.domain,
                doc_id=page.doc_id,
                file_name=page.file_name,
                page_index=page.page_index,
                text="",
                valid=False,
                invalid_reason=f"exception: {exc}",
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_extract, p): p for p in pending}
        for future in as_completed(futures):
            result: VLTableResult = future.result()
            completed += 1
            total_usage.add(result.usage)
            row = {
                "domain": result.domain,
                "doc_id": result.doc_id,
                "file_name": result.file_name,
                "page_index": result.page_index,
                "text": result.text,
                "valid": result.valid,
                "invalid_reason": result.invalid_reason,
                "usage": result.usage.to_dict(),
                "metadata": result.metadata,
            }
            append_jsonl(results_path, row)
            results.append(row)
            if not result.valid and result.invalid_reason != "not_a_table":
                failed += 1
            status = "OK" if result.valid else f"SKIP({result.invalid_reason})"
            print(f"  [{completed}/{len(pending)}] {result.domain}/{result.file_name} p{result.page_index + 1} {status}")

    print(f"[3/4] Extraction complete: {completed} done, {failed} failed")
    print(f"  Offline token: prompt={total_usage.prompt_tokens}, completion={total_usage.completion_tokens}, total={total_usage.total_tokens}")
    print(f"  (These tokens are NOT counted in final evaluation)")

    _write_summary(summary_path, b_pages, results, total_usage)
    print(f"[4/4] Summary written to {summary_path}")


def _write_summary(path: Path, b_pages, results, total_usage: TokenUsage) -> None:
    valid_count = sum(1 for r in results if r.get("valid"))
    invalid_reasons = Counter(r.get("invalid_reason", "") for r in results if not r.get("valid"))
    summary = {
        "b_class_total": len(b_pages),
        "extracted_total": len(results),
        "valid_tables": valid_count,
        "invalid_by_reason": dict(invalid_reasons),
        "offline_token_usage": total_usage.to_dict(),
        "note": "Offline Qwen-VL tokens are NOT counted in final evaluation token.",
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
