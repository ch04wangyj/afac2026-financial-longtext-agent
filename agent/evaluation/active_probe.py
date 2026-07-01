"""基于可信官网总分约束评估候选提交的上下界与诊断价值。"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from agent.evaluation.leaderboard_constraints import (
    LeaderboardRun,
    are_runs_feasible,
    infer_correctness_bounds,
)


@dataclass(frozen=True)
class ProbeEvaluation:
    """一个完整候选在当前隐藏标签版本空间中的可行结果。"""

    answers: dict[str, str]
    changes: tuple[tuple[str, str, str], ...]
    possible_correct_counts: tuple[int, ...]
    reference_correct_count: int

    @property
    def min_correct(self) -> int:
        return min(self.possible_correct_counts)

    @property
    def max_correct(self) -> int:
        return max(self.possible_correct_counts)

    @property
    def min_delta(self) -> int:
        return self.min_correct - self.reference_correct_count

    @property
    def max_delta(self) -> int:
        return self.max_correct - self.reference_correct_count

    @property
    def outcome_information_bits(self) -> float:
        """用等概率结果桶近似一次提交可提供的最大信息量。"""
        return math.log2(len(self.possible_correct_counts))

    def to_dict(self) -> dict:
        return {
            "changes": [
                {"qid": qid, "old_answer": old, "new_answer": new}
                for qid, old, new in self.changes
            ],
            "possible_correct_counts": list(self.possible_correct_counts),
            "min_correct": self.min_correct,
            "max_correct": self.max_correct,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
            "outcome_information_bits": round(self.outcome_information_bits, 6),
        }


def evaluate_probe(
    runs: list[LeaderboardRun],
    *,
    candidate_answers: dict[str, str],
    reference_answers: dict[str, str],
    reference_correct_count: int,
    valid_answers_by_qid: dict[str, set[str]],
) -> ProbeEvaluation:
    """计算候选可能获得的全部正确题数，而不是只报告乐观上界。"""
    bounds = infer_correctness_bounds(
        runs,
        baseline_answers=candidate_answers,
        valid_answers_by_qid=valid_answers_by_qid,
    )
    possible_counts = tuple(
        count
        for count in range(bounds.min_correct, bounds.max_correct + 1)
        if are_runs_feasible(
            [
                *runs,
                LeaderboardRun(
                    name="__probe__",
                    answers=candidate_answers,
                    correct_count=count,
                ),
            ],
            valid_answers_by_qid=valid_answers_by_qid,
        )
    )
    if not possible_counts:
        raise RuntimeError("候选没有任何与可信官网历史相容的正确题数")
    changes = tuple(
        (qid, reference_answers[qid], candidate_answers[qid])
        for qid in sorted(reference_answers)
        if reference_answers[qid] != candidate_answers[qid]
    )
    return ProbeEvaluation(
        answers=dict(candidate_answers),
        changes=changes,
        possible_correct_counts=possible_counts,
        reference_correct_count=reference_correct_count,
    )


def rank_probe_variants(
    runs: list[LeaderboardRun],
    *,
    reference_answers: dict[str, str],
    reference_correct_count: int,
    alternatives_by_qid: dict[str, tuple[str, ...]],
    valid_answers_by_qid: dict[str, set[str]],
    max_variants: int = 128,
) -> list[ProbeEvaluation]:
    """枚举小规模正交候选，优先保证最坏正确数，再比较潜在增益。"""
    qids = tuple(sorted(alternatives_by_qid))
    option_groups: list[tuple[str, ...]] = []
    for qid in qids:
        if qid not in reference_answers:
            raise KeyError(f"候选包含未知题目: {qid}")
        options = tuple(dict.fromkeys(alternatives_by_qid[qid]))
        invalid = sorted(set(options) - valid_answers_by_qid[qid])
        if invalid:
            raise ValueError(f"题目 {qid} 包含非法候选答案: {invalid}")
        option_groups.append(options)
    variant_count = math.prod(len(group) for group in option_groups)
    if variant_count > max_variants:
        raise ValueError(f"候选组合数 {variant_count} 超过上限 {max_variants}")

    evaluations: list[ProbeEvaluation] = []
    for selected in itertools.product(*option_groups):
        candidate = dict(reference_answers)
        candidate.update(dict(zip(qids, selected)))
        if candidate == reference_answers:
            continue
        evaluations.append(
            evaluate_probe(
                runs,
                candidate_answers=candidate,
                reference_answers=reference_answers,
                reference_correct_count=reference_correct_count,
                valid_answers_by_qid=valid_answers_by_qid,
            )
        )
    return sorted(
        evaluations,
        key=lambda item: (
            -item.min_correct,
            -item.max_correct,
            len(item.changes),
            -item.outcome_information_bits,
            item.changes,
        ),
    )
