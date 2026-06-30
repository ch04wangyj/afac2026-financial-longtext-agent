"""利用多次官网正确题数反推答案变更的可行性。

每次官网提交只暴露总正确题数，不暴露逐题标签。只要保存了各版本答案，
就可以把“每题真实答案只能取一个值”和“每个版本恰好命中 N 题”写成
0-1 整数约束。该模块只报告数学上可行或必然的结论，不用模型置信度替代标签。
"""

from __future__ import annotations

from dataclasses import dataclass


OTHER_ANSWER = "__other__"


@dataclass(frozen=True)
class LeaderboardRun:
    """一次已获得官网正确题数的完整提交。"""

    name: str
    answers: dict[str, str]
    correct_count: int


@dataclass(frozen=True)
class QuestionConstraintResult:
    """某题在全部官网约束下的可行答案集合。"""

    qid: str
    baseline_answer: str
    observed_answers: tuple[str, ...]
    feasible_answers: tuple[str, ...]
    baseline_forced_correct: bool
    baseline_forced_wrong: bool

    @property
    def forced_observed_answer(self) -> str:
        """仅当一个已观测答案被唯一确定时返回它。"""
        observed = [
            answer
            for answer in self.feasible_answers
            if answer != OTHER_ANSWER
        ]
        if len(self.feasible_answers) == 1 and len(observed) == 1:
            return observed[0]
        return ""

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "baseline_answer": self.baseline_answer,
            "observed_answers": list(self.observed_answers),
            "feasible_answers": list(self.feasible_answers),
            "baseline_forced_correct": self.baseline_forced_correct,
            "baseline_forced_wrong": self.baseline_forced_wrong,
            "forced_observed_answer": self.forced_observed_answer,
        }


@dataclass(frozen=True)
class CorrectnessBounds:
    """一组题目相对指定答案基线的可行正确数区间。"""

    question_count: int
    min_correct: int
    max_correct: int

    @property
    def min_wrong(self) -> int:
        return self.question_count - self.max_correct

    @property
    def max_wrong(self) -> int:
        return self.question_count - self.min_correct

    def to_dict(self) -> dict[str, int]:
        return {
            "question_count": self.question_count,
            "min_correct": self.min_correct,
            "max_correct": self.max_correct,
            "min_wrong": self.min_wrong,
            "max_wrong": self.max_wrong,
        }


def infer_weighted_assignment(
    runs: list[LeaderboardRun],
    *,
    baseline_name: str,
    valid_answers_by_qid: dict[str, set[str]],
    answer_weights: dict[str, dict[str, float]],
    partial_assignment: dict[str, str] | None = None,
) -> dict[str, str]:
    """在官网硬约束和已知标签内，求答案权重最大的完整标签配置。"""
    run_by_name = {run.name: run for run in runs}
    if baseline_name not in run_by_name:
        raise KeyError(f"未知基线运行: {baseline_name}")
    qids = _validate_runs(runs)
    states_by_qid: dict[str, tuple[str, ...]] = {}
    for qid in qids:
        states = tuple(sorted(valid_answers_by_qid[qid]))
        if not states:
            raise ValueError(f"题目 {qid} 没有合法答案")
        observed = {run.answers[qid] for run in runs}
        if not observed <= set(states):
            raise ValueError(f"题目 {qid} 的历史答案不在合法答案空间内")
        states_by_qid[qid] = states

    model = _build_model(runs, qids, states_by_qid)
    forced_variables = _validate_partial_assignment(
        partial_assignment or {},
        qids=qids,
        states_by_qid=states_by_qid,
        model=model,
    )
    objective = model.objective.copy()
    for (qid, answer), variable in model.variable_by_state.items():
        # SciPy 执行最小化，因此把共识支持分转为负目标值。
        objective[variable] = -float(
            answer_weights.get(qid, {}).get(answer, 0.0)
        )
        # 固定极小的字典序项，避免同分解在不同平台随机漂移。
        objective[variable] += variable * 1e-9
    solution = model.optimize(
        objective=objective,
        forced_variables=forced_variables,
    )
    if solution is None:
        raise RuntimeError("官网正确题数与提交答案之间不存在可行解")
    return {
        qid: next(
            answer
            for answer in states_by_qid[qid]
            if solution[model.variable_by_state[(qid, answer)]] > 0.5
        )
        for qid in qids
    }


def infer_correctness_bounds(
    runs: list[LeaderboardRun],
    *,
    baseline_answers: dict[str, str],
    valid_answers_by_qid: dict[str, set[str]],
    partial_assignment: dict[str, str] | None = None,
    subset_qids: set[str] | None = None,
) -> CorrectnessBounds:
    """计算一组题相对答案基线的最小和最大可行正确数。"""
    qids = _validate_runs(runs)
    if set(baseline_answers) != set(qids):
        raise ValueError("答案基线必须完整覆盖全部官网运行题目")
    selected_qids = set(qids) if subset_qids is None else set(subset_qids)
    unknown_subset = sorted(selected_qids - set(qids))
    if unknown_subset:
        raise KeyError(f"正确数区间包含未知题目: {unknown_subset}")

    states_by_qid: dict[str, tuple[str, ...]] = {}
    for qid in qids:
        states = tuple(sorted(valid_answers_by_qid[qid]))
        if baseline_answers[qid] not in states:
            raise ValueError(f"题目 {qid} 的基线答案非法: {baseline_answers[qid]}")
        observed = {run.answers[qid] for run in runs}
        if not observed <= set(states):
            raise ValueError(f"题目 {qid} 的历史答案不在合法答案空间内")
        states_by_qid[qid] = states

    model = _build_model(runs, qids, states_by_qid)
    forced_variables = _validate_partial_assignment(
        partial_assignment or {},
        qids=qids,
        states_by_qid=states_by_qid,
        model=model,
    )
    objective = model.objective.copy()
    for qid in selected_qids:
        objective[model.variable_by_state[(qid, baseline_answers[qid])]] = 1.0

    min_solution = model.optimize(
        objective=objective,
        forced_variables=forced_variables,
    )
    max_solution = model.optimize(
        objective=-objective,
        forced_variables=forced_variables,
    )
    if min_solution is None or max_solution is None:
        raise RuntimeError("官网正确题数、固定标签与提交答案之间不存在可行解")
    min_correct = int(round(float(objective @ min_solution)))
    max_correct = int(round(float(objective @ max_solution)))
    return CorrectnessBounds(
        question_count=len(selected_qids),
        min_correct=min_correct,
        max_correct=max_correct,
    )


def is_partial_assignment_feasible(
    runs: list[LeaderboardRun],
    *,
    valid_answers_by_qid: dict[str, set[str]],
    partial_assignment: dict[str, str],
) -> bool:
    """判断一组候选标签能否同时满足全部官网正确题数约束。"""
    qids = _validate_runs(runs)
    states_by_qid: dict[str, tuple[str, ...]] = {}
    for qid in qids:
        states = tuple(sorted(valid_answers_by_qid[qid]))
        if not states:
            raise ValueError(f"题目 {qid} 没有合法答案")
        observed = {run.answers[qid] for run in runs}
        if not observed <= set(states):
            raise ValueError(f"题目 {qid} 的历史答案不在合法答案空间内")
        states_by_qid[qid] = states
    model = _build_model(runs, qids, states_by_qid)
    forced_variables = _validate_partial_assignment(
        partial_assignment,
        qids=qids,
        states_by_qid=states_by_qid,
        model=model,
    )
    return model.solve(forced_variables=forced_variables)


def infer_question_constraints(
    runs: list[LeaderboardRun],
    *,
    baseline_name: str,
    valid_answers_by_qid: dict[str, set[str]] | None = None,
    partial_assignment: dict[str, str] | None = None,
) -> list[QuestionConstraintResult]:
    """在可选已知标签条件下枚举答案可行性，并返回相对基线的强制结论。"""
    if len(runs) < 2:
        raise ValueError("至少需要两次官网运行")
    run_by_name = {run.name: run for run in runs}
    if len(run_by_name) != len(runs):
        raise ValueError("运行名称必须唯一")
    if baseline_name not in run_by_name:
        raise KeyError(f"未知基线运行: {baseline_name}")

    qids = _validate_runs(runs)
    baseline = run_by_name[baseline_name]
    states_by_qid = {}
    for qid in qids:
        observed = {run.answers[qid] for run in runs}
        valid = (
            set(valid_answers_by_qid[qid])
            if valid_answers_by_qid is not None
            else set()
        )
        if valid and not observed <= valid:
            raise ValueError(f"题目 {qid} 的历史答案不在合法答案空间内")
        has_unobserved_valid_answer = not valid or bool(valid - observed)
        states_by_qid[qid] = tuple(
            [
                *sorted(observed),
                *([OTHER_ANSWER] if has_unobserved_valid_answer else []),
            ]
        )
    model = _build_model(runs, qids, states_by_qid)
    fixed_variables = _validate_partial_assignment(
        partial_assignment or {},
        qids=qids,
        states_by_qid=states_by_qid,
        model=model,
    )
    if not model.solve(forced_variables=fixed_variables):
        raise RuntimeError("官网正确题数与提交答案之间不存在可行解")

    output: list[QuestionConstraintResult] = []
    for qid in qids:
        feasible = tuple(
            answer
            for answer in states_by_qid[qid]
            if model.solve(
                forced_variable=model.variable_by_state[(qid, answer)],
                forced_variables=fixed_variables,
            )
        )
        baseline_answer = baseline.answers[qid]
        baseline_feasible = baseline_answer in feasible
        output.append(
            QuestionConstraintResult(
                qid=qid,
                baseline_answer=baseline_answer,
                observed_answers=tuple(
                    answer
                    for answer in states_by_qid[qid]
                    if answer != OTHER_ANSWER
                ),
                feasible_answers=feasible,
                baseline_forced_correct=feasible == (baseline_answer,),
                baseline_forced_wrong=not baseline_feasible,
            )
        )
    return output


def _validate_runs(runs: list[LeaderboardRun]) -> list[str]:
    """确保所有运行覆盖相同题目且正确题数合法。"""
    qids = sorted(runs[0].answers)
    expected = set(qids)
    if not qids:
        raise ValueError("运行答案不能为空")
    for run in runs:
        if set(run.answers) != expected:
            raise ValueError(f"运行 {run.name} 的题目集合与其他运行不一致")
        if not 0 <= run.correct_count <= len(qids):
            raise ValueError(f"运行 {run.name} 的正确题数非法")
    return qids


def _validate_partial_assignment(
    partial_assignment: dict[str, str],
    *,
    qids: list[str],
    states_by_qid: dict[str, tuple[str, ...]],
    model: "_MilpModel",
) -> list[int]:
    """校验已知标签并转换为需要固定为 1 的 MILP 变量。"""
    unknown = sorted(set(partial_assignment) - set(qids))
    if unknown:
        raise KeyError(f"候选包含未知题目: {unknown}")
    forced_variables: list[int] = []
    for qid, answer in partial_assignment.items():
        if answer not in states_by_qid[qid]:
            raise ValueError(f"题目 {qid} 的候选答案非法: {answer}")
        forced_variables.append(model.variable_by_state[(qid, answer)])
    return forced_variables


class _MilpModel:
    """对 SciPy MILP 的最小封装，便于重复执行单变量可行性检查。"""

    def __init__(
        self,
        *,
        objective,
        integrality,
        lower_bounds,
        upper_bounds,
        constraints,
        variable_by_state: dict[tuple[str, str], int],
    ) -> None:
        self.objective = objective
        self.integrality = integrality
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.constraints = constraints
        self.variable_by_state = variable_by_state

    def solve(
        self,
        forced_variable: int | None = None,
        forced_variables: list[int] | None = None,
    ) -> bool:
        """判断模型是否可行；可选地强制一个或多个题目状态取 1。"""
        return self.optimize(
            forced_variable=forced_variable,
            forced_variables=forced_variables,
        ) is not None

    def optimize(
        self,
        *,
        objective=None,
        forced_variable: int | None = None,
        forced_variables: list[int] | None = None,
    ):
        """执行一次整数优化并返回变量解；不可行时返回 ``None``。"""
        import numpy as np
        from scipy.optimize import Bounds, milp

        lower = np.array(self.lower_bounds, copy=True)
        upper = np.array(self.upper_bounds, copy=True)
        selected_variables = list(forced_variables or [])
        if forced_variable is not None:
            selected_variables.append(forced_variable)
        for variable in selected_variables:
            lower[variable] = 1.0
            upper[variable] = 1.0
        result = milp(
            c=self.objective if objective is None else objective,
            integrality=self.integrality,
            bounds=Bounds(lower, upper),
            constraints=self.constraints,
            options={"presolve": True},
        )
        return result.x if result.success else None


def _build_model(
    runs: list[LeaderboardRun],
    qids: list[str],
    states_by_qid: dict[str, tuple[str, ...]],
) -> _MilpModel:
    """构建“每题一个真实答案 + 每次提交命中数固定”的 0-1 模型。"""
    try:
        import numpy as np
        from scipy.optimize import LinearConstraint
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError(
            "排行榜约束审计需要 scipy>=1.9，请先安装 requirements.txt"
        ) from exc

    variable_by_state: dict[tuple[str, str], int] = {}
    for qid in qids:
        for answer in states_by_qid[qid]:
            variable_by_state[(qid, answer)] = len(variable_by_state)

    row_count = len(qids) + len(runs)
    matrix = lil_matrix((row_count, len(variable_by_state)), dtype=float)
    lower = np.zeros(row_count, dtype=float)
    upper = np.zeros(row_count, dtype=float)

    # 每道题必须且只能选择一个真实答案状态。
    for row_index, qid in enumerate(qids):
        for answer in states_by_qid[qid]:
            matrix[row_index, variable_by_state[(qid, answer)]] = 1.0
        lower[row_index] = 1.0
        upper[row_index] = 1.0

    # 每次官网运行命中的题数必须等于反推得到的整数正确题数。
    offset = len(qids)
    for run_index, run in enumerate(runs):
        row_index = offset + run_index
        for qid in qids:
            matrix[row_index, variable_by_state[(qid, run.answers[qid])]] = 1.0
        lower[row_index] = float(run.correct_count)
        upper[row_index] = float(run.correct_count)

    variable_count = len(variable_by_state)
    return _MilpModel(
        objective=np.zeros(variable_count, dtype=float),
        integrality=np.ones(variable_count, dtype=int),
        lower_bounds=np.zeros(variable_count, dtype=float),
        upper_bounds=np.ones(variable_count, dtype=float),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        variable_by_state=variable_by_state,
    )
