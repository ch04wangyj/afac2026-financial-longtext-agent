"""脚本 12：分析 Docling 样本并生成各领域规则草案。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Settings
from agent.preprocess.domain_rules import infer_candidate_rules, summarize_sample_bundle, top_signals


def main() -> None:
    settings = Settings.from_env()
    samples_root = settings.outputs_dir / 'docling_samples'
    if not samples_root.exists():
        raise RuntimeError(f'missing samples root: {samples_root}')

    summary_rows = []
    for domain_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        for sample_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            full_txt_path = sample_dir / 'full.txt'
            if not full_txt_path.exists():
                continue
            sample_text = full_txt_path.read_text(encoding='utf-8')
            rule_bundle = infer_candidate_rules(domain_dir.name, sample_text)
            stats = summarize_sample_bundle(sample_dir)
            signals = top_signals(sample_text)
            analysis = {
                'domain': domain_dir.name,
                'doc_id': sample_dir.name,
                'stats': stats,
                'candidate_rules': rule_bundle,
                'top_signals': signals,
            }
            (sample_dir / 'analysis.md').write_text(
                '# Docling Sample Analysis\n\n'
                f"- domain: {domain_dir.name}\n"
                f"- doc_id: {sample_dir.name}\n"
                f"- stats: {json.dumps(stats, ensure_ascii=False)}\n"
                f"- candidate_rules: {json.dumps(rule_bundle, ensure_ascii=False)}\n"
                f"- top_signals: {json.dumps(signals, ensure_ascii=False)}\n",
                encoding='utf-8',
            )
            summary_rows.append(analysis)
    out_path = settings.outputs_dir / 'docling_samples' / 'analysis_summary.json'
    out_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'wrote sample analysis summary -> {out_path}')


if __name__ == '__main__':
    main()
