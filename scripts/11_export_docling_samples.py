"""脚本 11：为每个领域导出一份或多份 Docling 样本。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.preprocess.docling_adapter import export_docling_sample_bundle
from agent.preprocess.profile import sample_output_dir


PDF_DOMAINS = ["insurance", "financial_contracts", "financial_reports", "research", "regulatory"]
MAX_SAMPLE_PAGES = {
    "insurance": 30,
    "financial_contracts": 180,
    "financial_reports": 240,
    "research": 80,
    "regulatory": 30,
}


def iter_sample_paths(domain_root: Path):
    for path in sorted(domain_root.rglob("*.pdf")):
        if path.is_file():
            yield path
    for path in sorted(domain_root.rglob("*.PDF")):
        if path.is_file():
            yield path


def candidate_score(path: Path) -> tuple[int, int, str]:
    page_count = estimate_pdf_page_count(path)
    return (page_count, path.stat().st_size, path.name)


def estimate_pdf_page_count(path: Path) -> int:
    import fitz

    with fitz.open(path) as doc:
        return len(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Docling sample bundles for one or more domains.")
    parser.add_argument("--domains", nargs="*", default=PDF_DOMAINS)
    parser.add_argument("--per-domain", type=int, default=1)
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()

    total = 0
    for domain in args.domains:
        domain_root = settings.raw_root / domain
        if not domain_root.exists():
            print(f"skip missing domain root: {domain_root}")
            continue

        candidates = []
        for path in iter_sample_paths(domain_root):
            try:
                page_count = estimate_pdf_page_count(path)
            except Exception as exc:
                print(f"skip unreadable pdf {path}: {exc}")
                continue
            if page_count > MAX_SAMPLE_PAGES.get(domain, 999999):
                continue
            candidates.append((candidate_score(path), path))
        candidates.sort(key=lambda item: item[0])

        exported = 0
        for _, path in candidates:
            out_dir = sample_output_dir(domain, path.stem, settings.outputs_dir)
            print(f"exporting docling sample {domain}/{path.stem} -> {out_dir}")
            export_docling_sample_bundle(path, out_dir)
            exported += 1
            total += 1
            if exported >= args.per_domain:
                break
        if exported == 0:
            print(f"no pdf sample found under {domain_root} within page threshold")
    print(f"exported {total} docling sample bundle(s)")


if __name__ == "__main__":
    main()
