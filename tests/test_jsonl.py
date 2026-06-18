"""JSONL 追加写入测试。"""

from pathlib import Path

from agent.io.jsonl import append_jsonl_rows, merge_rows_by_key, read_jsonl


def test_append_jsonl_rows_writes_all_rows_in_order(tmp_path):
    path = tmp_path / "out" / "rows.jsonl"
    rows = [{"id": "a", "text": "第一行"}, {"id": "b", "text": "第二行"}]

    count = append_jsonl_rows(path, rows)

    assert count == 2
    assert list(read_jsonl(path)) == rows


def test_merge_rows_by_key_preserves_requested_order():
    rows = [
        {"qid": "q2", "answer": "B"},
        {"qid": "q1", "answer": "A"},
        {"qid": "q2", "answer": "C"},
    ]

    merged = merge_rows_by_key(rows, key="qid", order=["q1", "q2"])

    assert merged == [
        {"qid": "q1", "answer": "A"},
        {"qid": "q2", "answer": "C"},
    ]
