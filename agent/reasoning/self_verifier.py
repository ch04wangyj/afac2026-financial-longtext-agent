"""V15 自验证迭代检索模块。

借鉴 FinAgent-RAG (arXiv 2605.05409) 的自验证迭代检索和 MARDoc (arXiv 2606.05749)
的结构化记忆，在首轮检索推理后做 3 项验证检查，REJECT 时自动 query refine 重检索。

不依赖人工核验，用 Qwen 做 verifier 做接受/拒绝决策。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object
from agent.schemas import Question, RetrievalResult, TokenUsage


@dataclass(frozen=True)
class SelfVerifierConfig:
    """自验证配置。"""

    max_iterations: int = 3
    confidence_threshold: float = 0.8  # 低于此值触发迭代
    verifier_max_tokens: int = 256
    # query refine 策略
    refine_strategies: tuple[str, ...] = (
        "synonym_expansion",  # 同义词扩展
        "broaden_scope",      # 扩大检索范围
        "alternative_path",   # 换检索路径
    )


@dataclass
class VerificationCheck:
    """单次验证检查结果。"""

    evidence_coverage: bool  # 证据是否覆盖所有选项
    key_values_supported: bool  # 关键数值是否有原文支撑
    reasoning_chain_complete: bool  # 推理链是否完整
    missing_items: list[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class IterativeRetrievalResult:
    """迭代检索的最终结果。"""

    accepted: bool
    final_answer: str
    final_confidence: float
    iterations: int
    refined_queries: list[str]
    verification_checks: list[VerificationCheck]
    total_usage: TokenUsage
    refined_context: str = ""


def build_verification_messages(
    question: Question,
    context: str,
    first_answer: str,
    first_confidence: float,
) -> list[dict[str, str]]:
    """构建自验证 prompt。

    让 Qwen 做 3 项检查，返回 ACCEPT 或 REJECT + 缺失项。
    """
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    return [
        {
            "role": "system",
            "content": (
                "你是金融答题自验证器。请检查首轮答案的可靠性，执行 3 项验证：\n"
                "1. 证据覆盖：是否每个选项都有至少 1 条证据？\n"
                "2. 数值支撑：关键数值是否有原文直接支撑（非推测）？\n"
                "3. 推理完整：推理链是否完整（无不确定/uncertain环节）？\n"
                "只有 3 项全部通过才 ACCEPT。返回紧凑 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题号：{question.qid}\n题干：{question.question}\n选项：\n{options}\n\n"
                f"首轮答案：{first_answer} (confidence={first_confidence})\n\n"
                f"证据上下文：\n{context[:6000]}\n\n"
                '返回紧凑 JSON：{"verdict":"ACCEPT|REJECT","checks":{"evidence_coverage":true,'
                '"key_values_supported":false,"reasoning_chain_complete":true},'
                '"missing":["选项C缺少数值证据"],"refine_query":"可选的新检索词"}'
            ),
        },
    ]


def build_query_refine_messages(
    question: Question,
    missing_items: list[str],
    strategy: str,
) -> list[dict[str, str]]:
    """构建 query refine prompt，生成新的检索词。"""
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    strategy_desc = {
        "synonym_expansion": "用同义词或近义词替换原检索词",
        "broaden_scope": "扩大检索范围，去掉过窄的限定词",
        "alternative_path": "换检索路径，从不同角度切入",
    }.get(strategy, "生成新检索词")

    return [
        {
            "role": "system",
            "content": f"你是金融检索查询优化器。{strategy_desc}。",
        },
        {
            "role": "user",
            "content": (
                f"题干：{question.question}\n选项：\n{options}\n\n"
                f"缺失项：{', '.join(missing_items)}\n\n"
                '返回紧凑 JSON：{"refined_queries":["查询1","查询2","查询3"]}'
            ),
        },
    ]


def run_self_verification(
    question: Question,
    context: str,
    first_answer: str,
    first_confidence: float,
    llm: QwenClient,
    config: SelfVerifierConfig | None = None,
    retrieve_fn=None,  # callable: (query, top_k) -> list[RetrievalResult]
) -> IterativeRetrievalResult:
    """执行自验证迭代检索。

    1. 首轮答案 conf < threshold 时启动
    2. Qwen 做 3 项验证
    3. REJECT 时 query refine + 重检索
    4. 最多 max_iterations 轮
    """
    cfg = config or SelfVerifierConfig()
    total_usage = TokenUsage()
    checks: list[VerificationCheck] = []
    refined_queries: list[str] = []
    current_context = context
    current_answer = first_answer
    current_confidence = first_confidence

    # 首轮就高置信度，直接接受
    if first_confidence >= cfg.confidence_threshold:
        return IterativeRetrievalResult(
            accepted=True,
            final_answer=first_answer,
            final_confidence=first_confidence,
            iterations=0,
            refined_queries=[],
            verification_checks=[],
            total_usage=total_usage,
            refined_context=context,
        )

    for iteration in range(cfg.max_iterations):
        # 验证
        verify_messages = build_verification_messages(
            question, current_context, current_answer, current_confidence
        )
        verify_response = llm.chat(
            verify_messages,
            temperature=0.0,
            max_tokens=cfg.verifier_max_tokens,
            enable_thinking=False,
        )
        total_usage.add(verify_response.usage)

        parsed = extract_json_object(verify_response.text) or {}
        verdict = str(parsed.get("verdict", "REJECT")).upper()
        check_data = parsed.get("checks", {})
        missing = parsed.get("missing", [])
        if isinstance(missing, str):
            missing = [missing]

        check = VerificationCheck(
            evidence_coverage=bool(check_data.get("evidence_coverage", False)),
            key_values_supported=bool(check_data.get("key_values_supported", False)),
            reasoning_chain_complete=bool(check_data.get("reasoning_chain_complete", False)),
            missing_items=missing,
            raw_response=verify_response.text,
        )
        checks.append(check)

        if verdict == "ACCEPT":
            return IterativeRetrievalResult(
                accepted=True,
                final_answer=current_answer,
                final_confidence=current_confidence,
                iterations=iteration + 1,
                refined_queries=refined_queries,
                verification_checks=checks,
                total_usage=total_usage,
                refined_context=current_context,
            )

        # REJECT: query refine
        if not retrieve_fn or not missing:
            break

        strategy = cfg.refine_strategies[iteration % len(cfg.refine_strategies)]
        refine_messages = build_query_refine_messages(question, missing, strategy)
        refine_response = llm.chat(
            refine_messages,
            temperature=0.0,
            max_tokens=cfg.verifier_max_tokens,
            enable_thinking=False,
        )
        total_usage.add(refine_response.usage)

        refine_parsed = extract_json_object(refine_response.text) or {}
        new_queries = refine_parsed.get("refined_queries", [])
        if isinstance(new_queries, str):
            new_queries = [new_queries]
        refined_queries.extend(new_queries)

        # 重检索
        if new_queries and retrieve_fn:
            new_evidence: list[RetrievalResult] = []
            for query in new_queries[:3]:
                results = retrieve_fn(query, top_k=8)
                new_evidence.extend(results)
            if new_evidence:
                # 追加到上下文（去重由调用方处理）
                new_text = "\n\n".join(
                    f"[补充证据] doc={e.doc_id} page={e.metadata.get('page')}\n{e.evidence_text[:300]}"
                    for e in new_evidence[:5]
                )
                current_context = f"{current_context}\n\n{new_text}"[:8000]

    # 超过最大轮次仍未 ACCEPT
    return IterativeRetrievalResult(
        accepted=False,
        final_answer=current_answer,
        final_confidence=current_confidence,
        iterations=len(checks),
        refined_queries=refined_queries,
        verification_checks=checks,
        total_usage=total_usage,
        refined_context=current_context,
    )
