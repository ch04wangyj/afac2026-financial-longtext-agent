"""V15 精确验证器：集成 PoT 数值推理、自验证迭代检索和自适应策略路由。

在 V14 precise_verifier 基础上增加：
1. 自适应策略路由：multi 题启用 thinking，mcq/tf 保持 no-thinking
2. PoT 数值推理：数值比较题生成受限 DSL 确定性执行
3. 自验证迭代检索：conf < 0.8 时自动 query refine 重检索
4. LLM-as-a-Judge 效用重排：BM25 Top-30 → Qwen Judge → Top-5

不引入 embedding，不依赖人工核验，用 Qwen 做 verifier/judge。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.index.bm25 import BM25SearchIndex
from agent.llm.qwen_client import QwenClient
from agent.reasoning.adaptive_router import (
    RouterConfig,
    llm_judge_rerank,
    route_question,
)
from agent.reasoning.answer_parser import extract_json_object, parse_answer
from agent.reasoning.pot_reasoner import (
    PoTConfig,
    format_pot_results,
    run_pot_reasoning,
)
from agent.reasoning.precise_verifier import (
    PreciseVerifier,
    PreciseVerifierConfig,
    _extract_confidence,
    _format_grouped_context,
    _valid_answer,
    build_precise_audit_messages,
    build_precise_judge_messages,
)
from agent.reasoning.self_verifier import (
    SelfVerifierConfig,
    run_self_verification,
)
from agent.retrieve.claims import build_claim_targets
from agent.retrieve.verification_queries import (
    extract_candidate_values,
    extract_predicate_terms,
)
from agent.retrieve.verification_rerank import select_verification_evidence
from agent.schemas import AnswerResult, Question, RetrievalResult, TokenUsage


@dataclass(frozen=True)
class V15VerifierConfig(PreciseVerifierConfig):
    """V15 配置，继承 V13/V14 并增加 PoT/自验证/路由参数。"""

    strategy_name: str = "v15_pot_verify"
    # 自验证
    enable_self_verification: bool = True
    self_verification_confidence_threshold: float = 0.8
    self_verification_max_iterations: int = 2  # 控制Token，最多2轮
    # PoT
    enable_pot: bool = True
    # LLM Judge
    enable_llm_judge: bool = True
    judge_top_k_input: int = 30
    judge_top_k_output: int = 8
    # 自适应路由
    enable_adaptive_routing: bool = True
    # thinking 默认关闭，由路由器按题型开启
    enable_thinking: bool = False


class V15PreciseVerifier(PreciseVerifier):
    """V15 精确验证器，集成 PoT + 自验证 + 自适应路由 + LLM Judge。"""

    def __init__(
        self,
        index: BM25SearchIndex,
        llm: QwenClient,
        config: V15VerifierConfig | None = None,
    ) -> None:
        super().__init__(index, llm, config or V15VerifierConfig())
        self.v15_config: V15VerifierConfig = config or V15VerifierConfig()

    def solve(self, question: Question) -> AnswerResult:
        """V15 solve：路由 → 检索 → Judge重排 → 推理 → PoT → 自验证。"""
        usage = TokenUsage()

        # 1. 自适应策略路由
        if self.v15_config.enable_adaptive_routing:
            from agent.reasoning.adaptive_router import RouterConfig, route_question

            router_cfg = RouterConfig()
            decision = route_question(question, router_cfg)
            enable_thinking = decision.enable_thinking
            max_tokens = decision.max_tokens
            routing_strategy = decision.strategy
        else:
            enable_thinking = self.v15_config.enable_thinking
            max_tokens = self.v15_config.answer_max_tokens
            routing_strategy = "default"

        # 2. 收集证据
        evidence, report = self.collect_evidence(question)

        # 3. LLM-as-a-Judge 效用重排
        if self.v15_config.enable_llm_judge and len(evidence) > self.v15_config.judge_top_k_output:
            evidence, judge_usage = llm_judge_rerank(
                question,
                evidence,
                self.llm,
                RouterConfig(
                    judge_top_k_input=self.v15_config.judge_top_k_input,
                    judge_top_k_output=self.v15_config.judge_top_k_output,
                ),
            )
            usage.add(judge_usage)

        context, evidence_map = _format_grouped_context(
            question, evidence, self.v15_config.max_context_chars
        )

        # 4. 首轮推理
        response = self.llm.chat(
            build_precise_judge_messages(question, context),
            temperature=0.0,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        usage.add(response.usage)
        final_text = response.text
        first_answer = parse_answer(final_text, question.answer_format)
        first_confidence = _extract_confidence(final_text)

        # 5. PoT 数值推理（如需要）
        pot_metadata = {}
        if self.v15_config.enable_pot:
            pot_result = run_pot_reasoning(
                question,
                evidence,
                context,
                self.llm,
                PoTConfig(),
            )
            if pot_result and pot_result.verified:
                usage.add(pot_result.usage)
                pot_text = format_pot_results(pot_result)
                if pot_text:
                    # 把 PoT 结果追加到上下文，重新推理
                    enhanced_context = f"{context}\n\n{pot_text}"
                    pot_response = self.llm.chat(
                        build_precise_judge_messages(question, enhanced_context),
                        temperature=0.0,
                        max_tokens=max_tokens,
                        enable_thinking=enable_thinking,
                    )
                    usage.add(pot_response.usage)
                    final_text = pot_response.text
                    first_answer = parse_answer(final_text, question.answer_format)
                    first_confidence = _extract_confidence(final_text)
                pot_metadata = {
                    "pot_executed": True,
                    "pot_programs": len(pot_result.executions),
                    "pot_raw": pot_result.raw_response[:500],
                }

        # 6. 自验证迭代检索（conf < threshold 时）
        verification_metadata = {}
        if self.v15_config.enable_self_verification and first_confidence < self.v15_config.self_verification_confidence_threshold:
            def _retrieve_fn(query: str, top_k: int = 8) -> list[RetrievalResult]:
                """供自验证模块重检索的回调。"""
                chunk_types = set(self.v15_config.search_chunk_types)
                doc_scope = list(dict.fromkeys(question.doc_ids))
                return self.index.search(
                    query,
                    top_k=top_k,
                    filter_doc_ids=set(doc_scope) or None,
                    filter_chunk_types=chunk_types,
                    source=f"{self.v15_config.strategy_name}:self_verify",
                    scoring_mode="bm25f_lite",
                )

            sv_result = run_self_verification(
                question,
                context,
                first_answer,
                first_confidence,
                self.llm,
                SelfVerifierConfig(
                    max_iterations=self.v15_config.self_verification_max_iterations,
                    confidence_threshold=self.v15_config.self_verification_confidence_threshold,
                ),
                retrieve_fn=_retrieve_fn,
            )
            usage.add(sv_result.total_usage)
            verification_metadata = {
                "self_verified": sv_result.accepted,
                "self_verify_iterations": sv_result.iterations,
                "self_verify_refined_queries": sv_result.refined_queries[:3],
            }
            # 如果自验证后上下文有补充，用增强上下文重新推理
            if sv_result.refined_context and sv_result.refined_context != context:
                final_response = self.llm.chat(
                    build_precise_judge_messages(question, sv_result.refined_context),
                    temperature=0.0,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
                usage.add(final_response.usage)
                final_text = final_response.text

        # 7. 答案解析
        answer = parse_answer(final_text, question.answer_format)
        if not _valid_answer(answer, question):
            answer = first_answer
        if not _valid_answer(answer, question):
            answer = sorted(question.options)[0]

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=_extract_confidence(final_text),
            evidence=evidence,
            token_usage=usage,
            raw_response=final_text,
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": self.v15_config.strategy_name,
                "model": self.llm.settings.qwen_model,
                "routing_strategy": routing_strategy,
                "enable_thinking": enable_thinking,
                "retrieval_report": report,
                "evidence_id_map": evidence_map,
                **pot_metadata,
                **verification_metadata,
            },
        )
