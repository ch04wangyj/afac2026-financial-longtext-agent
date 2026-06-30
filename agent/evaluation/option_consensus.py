"""比较多个独立运行的选项选择结果，生成候选审计而不自动改答案。"""

from __future__ import annotations

from dataclasses import dataclass

from agent.schemas import AnswerResult, Question


@dataclass(frozen=True)
class OptionConsensus:
    """一个选项在官网基线和多个候选运行之间的选择共识。"""

    qid: str
    option_key: str
    option_text: str
    baseline_selected: bool
    observed_runs: int
    opposite_votes: int
    same_votes: int
    flip_ratio: float
    unanimous_flip: bool
    run_votes: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "option_key": self.option_key,
            "option_text": self.option_text,
            "baseline_selected": self.baseline_selected,
            "observed_runs": self.observed_runs,
            "opposite_votes": self.opposite_votes,
            "same_votes": self.same_votes,
            "flip_ratio": round(self.flip_ratio, 6),
            "unanimous_flip": self.unanimous_flip,
            "run_votes": dict(self.run_votes),
        }


def audit_option_consensus(
    questions: list[Question],
    baseline_rows: list[AnswerResult],
    candidate_runs: dict[str, list[AnswerResult]],
    *,
    min_runs: int = 2,
) -> list[OptionConsensus]:
    """返回所有选项的运行共识；候选可为分域或部分题目结果。"""
    question_by_qid = {question.qid: question for question in questions}
    baseline_by_qid = {row.qid: row for row in baseline_rows}
    run_maps = {
        name: {row.qid: row for row in rows}
        for name, rows in candidate_runs.items()
    }
    output: list[OptionConsensus] = []
    for qid, baseline in baseline_by_qid.items():
        question = question_by_qid.get(qid)
        if question is None:
            continue
        for option_key, option_text in sorted(question.options.items()):
            baseline_selected = option_key in baseline.answer
            votes = {
                name: option_key in rows[qid].answer
                for name, rows in run_maps.items()
                if qid in rows
            }
            observed = len(votes)
            opposite = sum(value != baseline_selected for value in votes.values())
            same = observed - opposite
            ratio = opposite / observed if observed else 0.0
            output.append(
                OptionConsensus(
                    qid=qid,
                    option_key=option_key,
                    option_text=option_text,
                    baseline_selected=baseline_selected,
                    observed_runs=observed,
                    opposite_votes=opposite,
                    same_votes=same,
                    flip_ratio=ratio,
                    unanimous_flip=observed >= min_runs and opposite == observed,
                    run_votes=votes,
                )
            )
    return sorted(
        output,
        key=lambda row: (
            not row.unanimous_flip,
            -row.flip_ratio,
            -row.observed_runs,
            row.qid,
            row.option_key,
        ),
    )


def candidate_answer_from_consensus(
    question: Question,
    baseline_answer: str,
    rows: list[OptionConsensus],
) -> str:
    """仅把一致翻转应用到答案草案；调用方仍须人工核验后才能发布。"""
    selected = set(baseline_answer)
    for row in rows:
        if row.qid != question.qid or not row.unanimous_flip:
            continue
        if row.baseline_selected:
            selected.discard(row.option_key)
        else:
            selected.add(row.option_key)
    answer = "".join(sorted(selected))
    if question.answer_format in {"mcq", "tf"} and len(answer) != 1:
        return baseline_answer
    return answer or baseline_answer
