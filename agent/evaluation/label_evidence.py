"""区分语义证据与比赛隐藏标签证据，避免跨层推断。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LabelEvidenceLayer(str, Enum):
    """标签证据来源；只有前两类可以固定比赛隐藏标签。"""

    OFFICIAL_LABEL = "official_label"
    LEADERBOARD_FORCED = "leaderboard_forced"
    SOURCE_SEMANTIC = "source_semantic"
    MODEL_CONSENSUS = "model_consensus"


@dataclass(frozen=True)
class LabelEvidence:
    """一条带来源层级的答案证据。"""

    qid: str
    answer: str
    layer: LabelEvidenceLayer
    confirmed: bool
    source: str
    reason: str = ""


def build_benchmark_assignment(evidence: list[LabelEvidence]) -> dict[str, str]:
    """仅把官方标签或数学唯一标签转换为 MILP 固定条件。"""
    allowed_layers = {
        LabelEvidenceLayer.OFFICIAL_LABEL,
        LabelEvidenceLayer.LEADERBOARD_FORCED,
    }
    assignment: dict[str, str] = {}
    for item in evidence:
        if not item.confirmed:
            continue
        if item.layer not in allowed_layers:
            raise ValueError(
                f"{item.qid} 的 {item.layer.value} 只能支持语义判断，"
                "不能固定比赛隐藏标签"
            )
        if item.qid in assignment and assignment[item.qid] != item.answer:
            raise ValueError(f"题目 {item.qid} 存在冲突的隐藏标签证据")
        assignment[item.qid] = item.answer
    return assignment
