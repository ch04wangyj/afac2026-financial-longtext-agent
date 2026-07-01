"""读取并校验官网提交历史，防止重建快照污染隐藏标签推断。"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agent.evaluation.leaderboard_constraints import LeaderboardRun
from agent.io.jsonl import read_jsonl


VERIFIED_STATUS = "verified_submission"


@dataclass(frozen=True)
class LeaderboardRunRecord:
    """一次提交的元数据及其本地答案快照。"""

    name: str
    status: str
    usable_for_constraints: bool
    result_path: Path
    correct_count: int | None
    total_tokens: int | None
    official_score: float | None
    sha256: str
    answer_column: str = ""
    note: str = ""

    @property
    def is_verified(self) -> bool:
        """只有官网分数与本地文件一一对应的快照可进入硬约束。"""
        return self.status == VERIFIED_STATUS and self.usable_for_constraints


def load_run_registry(path: Path) -> list[LeaderboardRunRecord]:
    """加载注册表；相对路径以仓库根目录为基准。"""
    registry_path = path.resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("不支持的排行榜注册表版本")
    root = registry_path.parent.parent
    records: list[LeaderboardRunRecord] = []
    seen: set[str] = set()
    for row in payload.get("runs", []):
        name = str(row["name"])
        if name in seen:
            raise ValueError(f"排行榜注册表存在重复运行: {name}")
        seen.add(name)
        raw_path = Path(str(row["result_path"]))
        result_path = raw_path if raw_path.is_absolute() else root / raw_path
        records.append(
            LeaderboardRunRecord(
                name=name,
                status=str(row["status"]),
                usable_for_constraints=bool(row.get("usable_for_constraints", False)),
                result_path=result_path.resolve(),
                correct_count=(
                    int(row["correct_count"])
                    if row.get("correct_count") is not None
                    else None
                ),
                total_tokens=(
                    int(row["total_tokens"])
                    if row.get("total_tokens") is not None
                    else None
                ),
                official_score=(
                    float(row["official_score"])
                    if row.get("official_score") is not None
                    else None
                ),
                sha256=str(row.get("sha256", "")).lower(),
                answer_column=str(row.get("answer_column", "")),
                note=str(row.get("note", "")),
            )
        )
    return records


def load_verified_leaderboard_runs(
    path: Path,
    *,
    names: set[str] | None = None,
    verify_hashes: bool = True,
) -> list[LeaderboardRun]:
    """只加载明确标记为可信的官网提交，并校验文件哈希和题数。"""
    selected = [
        record
        for record in load_run_registry(path)
        if record.is_verified and (names is None or record.name in names)
    ]
    if names is not None:
        missing = sorted(names - {record.name for record in selected})
        if missing:
            raise KeyError(f"请求的运行未被注册为可信提交: {missing}")
    if len(selected) < 2:
        raise ValueError("排行榜硬约束至少需要两次可信提交")

    runs: list[LeaderboardRun] = []
    for record in selected:
        if not record.result_path.exists():
            raise FileNotFoundError(f"缺少提交快照: {record.result_path}")
        if verify_hashes:
            actual_hash = hashlib.sha256(record.result_path.read_bytes()).hexdigest()
            if not record.sha256 or actual_hash != record.sha256:
                raise RuntimeError(
                    f"提交快照哈希不匹配: {record.name}; "
                    f"expected={record.sha256}, actual={actual_hash}"
                )
        answers, token_total = _load_answers(record)
        if len(answers) != 100:
            raise ValueError(f"提交 {record.name} 不是完整 100 题快照")
        if (
            record.total_tokens is not None
            and token_total is not None
            and token_total != record.total_tokens
        ):
            raise RuntimeError(
                f"提交 {record.name} Token 不匹配: "
                f"expected={record.total_tokens}, actual={token_total}"
            )
        if record.correct_count is None:
            raise ValueError(f"可信提交 {record.name} 缺少正确题数")
        runs.append(
            LeaderboardRun(
                name=record.name,
                answers=answers,
                correct_count=record.correct_count,
            )
        )
    return runs


def _load_answers(
    record: LeaderboardRunRecord,
) -> tuple[dict[str, str], int | None]:
    """读取完整 JSONL 结果或可提交到 Git 的紧凑答案矩阵。"""
    if record.answer_column:
        answers: dict[str, str] = {}
        with record.result_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or record.answer_column not in reader.fieldnames:
                raise ValueError(
                    f"答案矩阵缺少列 {record.answer_column}: {record.result_path}"
                )
            for row in reader:
                qid = str(row.get("qid", "")).strip()
                if not qid:
                    continue
                if qid in answers:
                    raise ValueError(f"提交 {record.name} 存在重复题号: {qid}")
                answers[qid] = str(row[record.answer_column]).strip()
        return answers, None

    answers = {}
    token_total = 0
    for row in read_jsonl(record.result_path):
        qid = str(row.get("qid", ""))
        if not qid or qid == "summary":
            continue
        if qid in answers:
            raise ValueError(f"提交 {record.name} 存在重复题号: {qid}")
        answers[qid] = str(row["answer"])
        token_total += int(row.get("token_usage", {}).get("total_tokens", 0))
    return answers, token_total
