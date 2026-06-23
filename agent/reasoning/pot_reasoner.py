"""V15 Program-of-Thought 数值推理模块。

借鉴 FinAgent-RAG (arXiv 2605.05409) 的 PoT 和 DCRC (KDD 2026) 的编译执行推理，
让 Qwen 生成受限 DSL 代码做精确算术，而非依赖模型心算。

DSL 仅允许四种操作：compare / difference / ratio / growth_rate
操作数必须绑定 fact_ledger 中的已抽取事实，不允任意代码执行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object
from agent.reasoning.fact_ledger import compile_numeric_fact_ledger, format_numeric_fact_ledger
from agent.schemas import Question, RetrievalResult, TokenUsage


# 比较谓词检测
COMPARISON_PREDICATES = (
    "高于",
    "低于",
    "超过",
    "不足",
    "大于",
    "小于",
    "同比",
    "环比",
    "占比",
    "增长率",
    "增速",
    "增幅",
    "下降",
    "增长",
    "提升",
    "降低",
    "减少",
    "增加",
    "比例",
    "倍",
    "一半",
    "十分之一",
)

# 受限 DSL 允许的操作
ALLOWED_OPS = {"compare", "difference", "ratio", "growth_rate"}


@dataclass(frozen=True)
class PoTConfig:
    """PoT 推理配置。"""

    max_program_tokens: int = 512
    max_execution_results: int = 8
    enable_for_answer_formats: tuple[str, ...] = ("multi", "tf")
    confidence_threshold: float = 0.8  # 仅 conf < 此值时触发 PoT


@dataclass
class PoTResult:
    """PoT 推理结果。"""

    program: str
    executions: list[dict[str, Any]]
    verified: bool
    usage: TokenUsage
    raw_response: str = ""


def needs_pot(question: Question) -> bool:
    """检测题目是否需要 PoT 数值推理。

    触发条件：
    1. 题型为 multi 或 tf
    2. 题干或选项包含比较谓词
    3. 题干包含数值比较语义
    """
    text = f"{question.question} {' '.join(question.options.values())}"
    has_predicate = any(pred in text for pred in COMPARISON_PREDICATES)
    has_number = bool(re.search(r"\d+", text))
    return has_predicate and has_number


def build_pot_messages(
    question: Question,
    context: str,
    fact_ledger_text: str,
) -> list[dict[str, str]]:
    """构建 PoT 推理的 prompt。

    让 Qwen 生成受限 DSL 代码，而非直接给答案。
    """
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是金融数值推理引擎。请基于证据和数值事实账本，为每个需要数值比较的选项生成受限 DSL 程序。\n"
                "DSL 仅允许以下四种操作：\n"
                "  compare(fact_a, fact_b) → 返回 'greater' | 'less' | 'equal' | 'incomparable'\n"
                "  difference(fact_a, fact_b) → 返回差值（同一单位）\n"
                "  ratio(fact_a, fact_b) → 返回比值\n"
                "  growth_rate(fact_current, fact_previous) → 返回增长率百分比\n"
                "操作数必须是事实账本中的 fact_id（如 F1_R1），不允许字面数值。\n"
                "如果某选项不需要数值计算，标注 skip。\n"
                "返回紧凑 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题干：{question.question}\n选项：\n{options}\n\n"
                f"证据：\n{context}\n\n"
                f"数值事实账本：\n{fact_ledger_text}\n\n"
                "为每个选项生成 DSL 程序，返回紧凑 JSON：\n"
                '{"programs":{"A":{"dsl":"compare(F1_R1, F2_R1)","reason":"比较营收"},"B":{"dsl":"skip","reason":"非数值题"}},'
                '"executions_needed":["A"],"answer":"A","confidence":0.9}'
            ),
        },
    ]


def run_pot_reasoning(
    question: Question,
    evidence: list[RetrievalResult],
    context: str,
    llm: QwenClient,
    config: PoTConfig | None = None,
) -> PoTResult | None:
    """执行 PoT 数值推理。

    1. 编译数值事实账本
    2. 让 Qwen 生成受限 DSL 程序
    3. 确定性执行 DSL
    4. 返回执行结果
    """
    if not needs_pot(question):
        return None

    cfg = config or PoTConfig()
    ledger = compile_numeric_fact_ledger(question, evidence)
    fact_text = format_numeric_fact_ledger(ledger)
    if "无可验证数值事实" in fact_text:
        return None

    messages = build_pot_messages(question, context, fact_text)
    response = llm.chat(
        messages,
        temperature=0.0,
        max_tokens=cfg.max_program_tokens,
        enable_thinking=False,
    )

    parsed = extract_json_object(response.text)
    if not parsed:
        return PoTResult(
            program="",
            executions=[],
            verified=False,
            usage=response.usage,
            raw_response=response.text,
        )

    programs = parsed.get("programs", {})
    executions: list[dict[str, Any]] = []
    facts_by_id = {f["fact_id"]: f for f in ledger.get("facts", [])}

    for option_key, program_info in programs.items():
        dsl = str(program_info.get("dsl", "")).strip()
        if dsl == "skip" or not dsl:
            continue
        result = _execute_dsl(dsl, facts_by_id)
        if result is not None:
            executions.append({
                "option": option_key,
                "dsl": dsl,
                "result": result,
                "reason": program_info.get("reason", ""),
            })

    return PoTResult(
        program=response.text,
        executions=executions,
        verified=len(executions) > 0,
        usage=response.usage,
        raw_response=response.text,
    )


def _execute_dsl(dsl: str, facts_by_id: dict[str, dict]) -> Any:
    """确定性执行受限 DSL。

    只允许 compare/difference/ratio/growth_rate 四种操作，
    操作数必须是 facts_by_id 中的 key。
    """
    dsl = dsl.strip()
    # 解析 operation(fact_a, fact_b)
    match = re.match(r"(\w+)\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", dsl)
    if not match:
        return None

    op, fact_a_id, fact_b_id = match.groups()
    if op not in ALLOWED_OPS:
        return None

    fact_a = facts_by_id.get(fact_a_id)
    fact_b = facts_by_id.get(fact_b_id)
    if not fact_a or not fact_b:
        return None

    val_a = _to_decimal(fact_a.get("normalized_value", ""))
    val_b = _to_decimal(fact_b.get("normalized_value", ""))
    if val_a is None or val_b is None:
        return None

    try:
        if op == "compare":
            if val_a > val_b:
                return "greater"
            elif val_a < val_b:
                return "less"
            else:
                return "equal"
        elif op == "difference":
            return str(val_a - val_b)
        elif op == "ratio":
            if val_b == 0:
                return "incomparable"
            return str(val_a / val_b)
        elif op == "growth_rate":
            if val_b == 0:
                return "incomparable"
            rate = (val_a - val_b) / val_b * Decimal("100")
            return f"{rate:.2f}%"
    except (InvalidOperation, ZeroDivisionError):
        return None

    return None


def _to_decimal(value: str) -> Decimal | None:
    """安全转换字符串为 Decimal。"""
    try:
        return Decimal(str(value).replace(",", "").replace("，", "").strip())
    except (InvalidOperation, ValueError):
        return None


def format_pot_results(pot_result: PoTResult) -> str:
    """把 PoT 执行结果格式化为可追加到上下文的文本。"""
    if not pot_result.executions:
        return ""
    lines = ["[PoT数值验证结果]"]
    for exec_result in pot_result.executions:
        lines.append(
            f"  选项{exec_result['option']}: {exec_result['dsl']} → {exec_result['result']}"
            f" ({exec_result['reason']})"
        )
    return "\n".join(lines)
