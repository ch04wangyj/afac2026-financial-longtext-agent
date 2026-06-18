from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE_MOD = _load_module(
    "compare_runs_script",
    Path("D:/pyproject/afac2026-financial-longtext-agent/scripts/09_compare_runs.py"),
)
REPORT_MOD = _load_module(
    "report_results_script",
    Path("D:/pyproject/afac2026-financial-longtext-agent/scripts/08_report_results.py"),
)
DELTA_MOD = _load_module(
    "diagnose_answer_delta_script",
    Path("D:/pyproject/afac2026-financial-longtext-agent/scripts/diagnose_answer_delta.py"),
)



def test_compare_runs_counts_changes_and_token_delta(tmp_path):
    base = tmp_path / 'base.jsonl'
    cand = tmp_path / 'cand.jsonl'
    base.write_text('{"qid":"q1","answer":"A","token_usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"evidence":[{"doc_id":"d1"}]}\n', encoding='utf-8')
    cand.write_text('{"qid":"q1","answer":"AB","token_usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6},"evidence":[{"doc_id":"d2"}]}\n', encoding='utf-8')
    report = COMPARE_MOD.compare_runs(COMPARE_MOD._load(base), COMPARE_MOD._load(cand))
    assert report['data']['question_count'] == 1
    assert report['data']['answer_change_count'] == 1
    assert report['data']['delta']['total_tokens'] == 3
    assert report['data']['rows'][0]['baseline_docs'] == ['d1']
    assert report['data']['rows'][0]['candidate_docs'] == ['d2']



def test_compare_answer_results_counts_changed_answers(tmp_path):
    base = tmp_path / 'base.jsonl'
    cand = tmp_path / 'cand.jsonl'
    base.write_text('{"qid":"q1","answer":"A","metadata":{"domain":"research","answer_format":"multi"},"token_usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3},"evidence":[{"doc_id":"d1"}]}\n', encoding='utf-8')
    cand.write_text('{"qid":"q1","answer":"AB","metadata":{"domain":"research","answer_format":"multi"},"token_usage":{"prompt_tokens":2,"completion_tokens":4,"total_tokens":6},"evidence":[{"doc_id":"d2"}]}\n', encoding='utf-8')
    report = DELTA_MOD.compare_answer_results(base, cand)
    assert report['data']['question_count'] == 1
    assert report['data']['answer_change_count'] == 1
    assert report['data']['by_domain']['research'] == 1



def test_build_report_marks_missing_docs_and_none_options(tmp_path):
    rows = [
        {
            'qid': 'q1',
            'answer': 'A',
            'confidence': 0.8,
            'token_usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            'evidence': [{'doc_id': 'd1'}],
            'metadata': {
                'answer_format': 'multi',
                'strategy': 'option_matrix',
                'option_judgements': [{'option': 'A', 'verdict': None}],
            },
        }
    ]
    manifest = {'q1': {'qid': 'q1', 'domain': 'financial_reports', 'answer_format': 'multi', 'doc_ids': ['d1', 'd2']}}
    report = REPORT_MOD.build_report(rows, manifest)
    assert report['data']['issues'][0]['missing_docs'] == ['d2']
    assert report['data']['issues'][0]['none_options'] == ['A']
