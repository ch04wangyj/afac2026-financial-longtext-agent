"""单题求解器：串联检索、压缩、Qwen 作答和答案解析。"""

from __future__ import annotations

import re

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.llm.qwen_client import QwenClient
from agent.reasoning import logicrag, multi_logicrag
from agent.reasoning.answer_parser import extract_json_object, parse_answer, parse_verdict
from agent.reasoning.claim_verifier import (
    analyze_claim_evidence_sufficiency,
    assemble_claim_answer,
    build_claim_refinement,
    should_refine_claim,
)
from agent.reasoning.option_matrix import OptionVerdict, parse_option_verdict, synthesize_answer
from agent.reasoning.retrieval_refiner import build_lightweight_refined_queries, should_trigger_retrieval_refinement
from agent.reasoning.prompts import (
    build_answer_messages,
    build_financial_metric_extraction_messages,
    build_logicrag_final_compose_messages,
    build_option_evidence_judgement_messages,
    build_option_judgement_messages,
)
from agent.runtime.logicrag_config import load_logicrag_runtime_config
from agent.runtime.parallel import parallel_map_ordered
from agent.domain.coverage_rules import expected_evidence_facets
from agent.retrieve.claim_retrieval import build_claim_query_bundles, retrieve_claim_candidates
from agent.retrieve.claims import build_claim_targets, claim_to_retrieval_target
from agent.retrieve.coverage import assess_doc_coverage, retrieve_missing_doc_evidence
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.option_retrieval import build_option_queries, retrieve_option_candidates
from agent.retrieve.rerank import rerank_retrieval_results
from agent.retrieve.retriever import Retriever
from agent.retrieve.targets import analyze_evidence_sufficiency, build_retrieval_target
from agent.schemas import AnswerResult, Question, RetrievalResult, TokenUsage


class Solver:
    """Agent 的最小闭环执行单元。"""

    def __init__(self, retriever: Retriever, compressor: RuleEvidenceCompressor, llm: QwenClient) -> None:
        self.retriever = retriever
        self.compressor = compressor
        self.llm = llm
        self.option_compressors: dict[str, RuleEvidenceCompressor] = {}
        self.runtime = load_logicrag_runtime_config()

    def solve(self, question: Question) -> AnswerResult:
        """处理一道题并返回可提交答案与可审计证据。"""
        if (
            question.answer_format == "multi"
            and self.llm.settings.retrieval_strategy == "logicrag_agent"
            and self.llm.settings.logicrag_enabled
            and self.runtime.a_board.claim_centric_multi_enabled
        ):
            return self._solve_claim_centric(question)
        if (
            question.answer_format == "mcq"
            and self.llm.settings.retrieval_strategy == "logicrag_agent"
            and self.llm.settings.logicrag_enabled
            and self.runtime.a_board.claim_centric_mcq_enabled
        ):
            return self._solve_claim_centric(question)
        if (
            question.answer_format == "multi"
            and self.llm.settings.retrieval_strategy == "logicrag_agent"
            and self.llm.settings.logicrag_enabled
            and self.runtime.a_board.multi_logicrag_enabled
        ):
            return self._solve_multi_logicrag(question)
        if self.llm.settings.retrieval_strategy == "logicrag_agent" and self.llm.settings.logicrag_enabled:
            return self._solve_logicrag_agent(question)
        if question.answer_format == "multi" and self.runtime.a_board.option_matrix_enabled:
            return self._solve_by_option_matrix(question)
        if question.answer_format == "multi" and self.llm.settings.enable_multi_option_judgement:
            return self._solve_multi_by_option(question)

        answer_profile = self.llm.settings.thinking_profile_for_step("answer_single_pass")
        retrieved = self.retriever.retrieve(question)
        evidence = self.compressor.compress(question, retrieved)
        evidence, coverage_report = self._apply_a_board_coverage_gate(question, retrieved, evidence)
        messages = build_answer_messages(question, evidence)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            thinking_profile=answer_profile,
        )
        answer = parse_answer(response.text, question.answer_format)
        confidence = _extract_confidence(response.text)
        if not answer:
            # 格式解析失败时给出可提交兜底值，同时把置信度降为 0，便于后续复核。
            answer = "A"
            confidence = 0.0
        financial_metric_extraction = self._maybe_extract_financial_metrics(question, evidence, response.usage)
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=evidence,
            token_usage=response.usage,
            raw_response=response.text,
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "reasoning": response.reasoning,
                "coverage_report": coverage_report,
                "financial_metric_extraction": financial_metric_extraction,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
            },
        )

    def _solve_logicrag_agent(self, question: Question) -> AnswerResult:
        """显式执行 LogicRAG agent：规划 -> rank-wise 检索/记忆 -> compose。"""
        total_usage = TokenUsage()
        plan, plan_response = logicrag.plan_logic_subproblems_with_qwen(
            question,
            self.llm,
            max_subproblems=self.llm.settings.logicrag_max_subproblems,
            max_ranks=self.llm.settings.logicrag_max_ranks,
        )
        total_usage.add(plan_response.usage)

        rank_memories: list[dict] = []
        if self.runtime.logicrag.adaptive_retrieval_enabled:
            rank_runs, combined, adaptive_usage = logicrag.retrieve_rankwise_evidence_adaptive(
                self.retriever,
                question,
                plan,
                self.llm,
                self.runtime,
                per_query_top_k=max(1, min(self.retriever.top_k_per_query, self.runtime.logicrag.rank_top_k)),
                fused_top_k=self.runtime.logicrag.rank_top_k,
            )
            total_usage.add(adaptive_usage)
        else:
            rank_runs, combined = logicrag.retrieve_rankwise_evidence(
                self.retriever,
                question,
                plan,
                per_query_top_k=max(1, min(self.retriever.top_k_per_query, self.runtime.logicrag.rank_top_k)),
                fused_top_k=self.runtime.logicrag.rank_top_k,
            )

        for run in rank_runs:
            response = logicrag.summarize_rank_memory_with_qwen(
                question,
                self.llm,
                rank=run["rank"],
                nodes=run["nodes"],
                evidence=run["results"],
                prior_memories=rank_memories,
                max_chars=self.llm.settings.logicrag_memory_chars,
            )
            total_usage.add(response.usage)
            rank_memories.append(
                {
                    "rank": run["rank"],
                    "summary": response.text,
                    "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in run["results"]))[:3],
                }
            )

        final_compose_profile = self.llm.settings.thinking_profile_for_step("logicrag_final_compose")
        audit_evidence = self.compressor.compress(question, combined)
        audit_evidence, coverage_report = self._apply_a_board_coverage_gate(question, combined, audit_evidence)
        compose_source = (rank_runs[-1].get("compose_results") or rank_runs[-1]["results"]) if rank_runs else combined
        compose_evidence = self.compressor.compress(question, compose_source)
        messages = build_logicrag_final_compose_messages(question, compose_evidence, plan, rank_memories)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            thinking_profile=final_compose_profile,
        )
        total_usage.add(response.usage)

        answer = parse_answer(response.text, question.answer_format)
        confidence = _extract_confidence(response.text)
        if not answer:
            answer = "A"
            confidence = 0.0
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=audit_evidence,
            token_usage=total_usage,
            raw_response=response.text,
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "logicrag_agent",
                "reasoning": response.reasoning,
                "logic_plan": plan.to_dict(),
                "plan_token_usage": plan_response.usage.to_dict(),
                "rank_memories": rank_memories,
                "rank_runs": [
                    {
                        "rank": run["rank"],
                        "queries": list(run.get("queries", [])),
                        "sufficiency": dict(run.get("sufficiency", {})),
                        "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in run.get("results", []))),
                        "adaptive_retrieval": dict(run.get("adaptive_retrieval", {})),
                    }
                    for run in rank_runs
                ],
                "final_compose_evidence_scope": "last_rank_only",
                "coverage_report": coverage_report,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
            },
        )

    def _solve_claim_centric(self, question: Question) -> AnswerResult:
        total_usage = TokenUsage()
        filter_doc_ids = self.retriever._candidate_doc_filter(question, not self.runtime.a_board.use_doc_ids_as_hint_only)
        claims = build_claim_targets(question)
        shared_query = self.retriever._question_with_options(question) if hasattr(self.retriever, "_question_with_options") else question.question
        shared_candidates = self.retriever.index.search(
            shared_query,
            top_k=self.runtime.a_board.max_option_candidates,
            filter_doc_ids=filter_doc_ids,
            source="claim_shared",
        )
        claim_runs: dict[str, dict] = {}
        verdicts: dict[str, dict] = {}
        all_evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()
        threshold = self.runtime.a_board.low_confidence_threshold
        verdict_profile = self.llm.settings.thinking_profile_for_step("multi_logicrag_option_verdict")
        retry_profile = self.llm.settings.thinking_profile_for_step("multi_logicrag_option_retry")

        for claim in claims:
            option_question = multi_logicrag.build_multi_option_question(question, claim.option_key, claim.option_text)
            candidates, bundles = retrieve_claim_candidates(
                self.retriever.index,
                question,
                claim,
                filter_doc_ids=filter_doc_ids,
                top_k_per_query=self.retriever.top_k_per_query,
                fused_top_k=self.runtime.a_board.max_option_candidates,
                shared_candidates=shared_candidates,
            )
            reranked = rerank_retrieval_results(question, claim_to_retrieval_target(claim), candidates, top_k=self.runtime.a_board.max_option_candidates)
            evidence = self._option_compressor(question.domain).compress(option_question, reranked)
            coverage = assess_doc_coverage(question.doc_ids, evidence).to_dict() if question.doc_ids else {}
            sufficiency = analyze_claim_evidence_sufficiency(claim, [item.evidence_text for item in evidence])
            verdict_evidence = evidence[: self.runtime.a_board.claim_verdict_max_evidence_items]
            response = self.llm.chat(
                build_option_evidence_judgement_messages(question, claim.option_key, claim.option_text, verdict_evidence),
                temperature=0.0,
                thinking_profile=verdict_profile,
            )
            total_usage.add(response.usage)
            parsed = parse_option_verdict(response.text, claim.option_key)
            verdict = {
                "relation": _relation_from_option_verdict(parsed),
                "confidence": float(parsed.confidence or 0.0),
                "support_evidence": list(parsed.support_evidence),
                "refute_evidence": list(parsed.refute_evidence),
                "reason": parsed.reason,
                "raw_response": parsed.raw_response,
            }
            retried = False
            refinement = {"action": "", "reason": "", "queries": []}
            retry_queries: list[str] = []
            if self.runtime.a_board.multi_logicrag_retry_enabled and should_refine_claim(sufficiency, verdict, threshold=threshold):
                retried = True
                claim_refinement = build_claim_refinement(claim, sufficiency)
                refinement = claim_refinement.to_dict()
                retry_queries = list(claim_refinement.queries)[: self.runtime.a_board.max_claim_query_bundles]
                ranked_lists = [
                    self.retriever.index.search(
                        query=query,
                        top_k=self.retriever.top_k_per_query + max(2, self.runtime.a_board.max_verifier_candidates_per_option // 2),
                        filter_doc_ids=filter_doc_ids,
                        source=f"claim_retry_{claim.option_key}",
                    )
                    for query in retry_queries
                ]
                expanded = reciprocal_rank_fusion(ranked_lists, top_k=self.runtime.a_board.max_option_candidates + self.runtime.a_board.max_verifier_candidates_per_option) if ranked_lists else []
                merged = multi_logicrag.merge_unique_evidence(reranked, expanded)
                reranked = rerank_retrieval_results(question, claim_to_retrieval_target(claim), merged, top_k=self.runtime.a_board.max_option_candidates)
                evidence = self._option_compressor(question.domain).compress(option_question, reranked)
                coverage = assess_doc_coverage(question.doc_ids, evidence).to_dict() if question.doc_ids else {}
                sufficiency = analyze_claim_evidence_sufficiency(claim, [item.evidence_text for item in evidence])
                retry_verdict_evidence = evidence[: self.runtime.a_board.claim_retry_verdict_max_evidence_items]
                retry_response = self.llm.chat(
                    build_option_evidence_judgement_messages(question, claim.option_key, claim.option_text, retry_verdict_evidence),
                    temperature=0.0,
                    thinking_profile=retry_profile,
                )
                total_usage.add(retry_response.usage)
                parsed = parse_option_verdict(retry_response.text, claim.option_key)
                verdict = {
                    "relation": _relation_from_option_verdict(parsed),
                    "confidence": float(parsed.confidence or 0.0),
                    "support_evidence": list(parsed.support_evidence),
                    "refute_evidence": list(parsed.refute_evidence),
                    "reason": parsed.reason,
                    "raw_response": parsed.raw_response,
                }

            for item in evidence:
                if item.chunk_id not in seen_chunks:
                    all_evidence.append(item)
                    seen_chunks.add(item.chunk_id)
            verdicts[claim.option_key] = verdict
            claim_runs[claim.option_key] = {
                "claim_id": claim.claim_id,
                "option_key": claim.option_key,
                "claim_type": claim.claim_type,
                "query_bundles": [bundle.to_dict() for bundle in bundles],
                "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in evidence)),
                "evidence_chunk_ids": [item.chunk_id for item in evidence],
                "coverage": coverage,
                "sufficiency": sufficiency,
                "retried": retried,
                "refinement": refinement,
                "retry_queries": retry_queries,
                "relation": verdict["relation"],
                "confidence": verdict["confidence"],
                "support_evidence": verdict["support_evidence"],
                "refute_evidence": verdict["refute_evidence"],
                "reason": verdict["reason"],
            }

        answer = assemble_claim_answer(verdicts, answer_format=question.answer_format)
        confidence = _average_known_confidence(list(verdicts.values()))
        final_evidence = _limit_evidence_with_doc_coverage(all_evidence, self.compressor.top_k, question.doc_ids)
        final_evidence, coverage_report = self._apply_a_board_coverage_gate(question, all_evidence, final_evidence)
        if not answer:
            if question.answer_format == "multi":
                answer = _fallback_multi_option([
                    {"option": key, "verdict": None if value["relation"] == "insufficient" else value["relation"] == "support", "confidence": value["confidence"]}
                    for key, value in verdicts.items()
                ]) or "A"
                confidence = 0.0
            else:
                answer = sorted(question.options)[0] if question.options else "A"
                confidence = 0.0
        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=final_evidence,
            token_usage=total_usage,
            raw_response="",
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "claim_centric_multi" if question.answer_format == "multi" else "claim_centric_mcq",
                "answer_assembly_policy": "multi_all_supported" if question.answer_format == "multi" else "single_strongest_supported",
                "claim_runs": claim_runs,
                "token_budget_policy": {
                    "max_claim_query_bundles": self.runtime.a_board.max_claim_query_bundles,
                    "max_claim_refinement_rounds": self.runtime.a_board.max_claim_refinement_rounds,
                    "claim_final_compose_enabled": self.runtime.a_board.claim_final_compose_enabled,
                    "claim_verdict_max_evidence_items": self.runtime.a_board.claim_verdict_max_evidence_items,
                    "claim_retry_verdict_max_evidence_items": self.runtime.a_board.claim_retry_verdict_max_evidence_items,
                },
                "shared_retrieval": {
                    "queries": [shared_query],
                    "candidate_count": len(shared_candidates),
                    "doc_ids": list(dict.fromkeys(item.doc_id for item in shared_candidates)),
                },
                "coverage_report": coverage_report,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
            },
        )

    def _solve_multi_logicrag(self, question: Question) -> AnswerResult:
        """多选题强化 LogicRAG：逐选项检索/判定，对每个不确定选项执行一轮扩检。"""
        total_usage = TokenUsage()
        filter_doc_ids = self.retriever._candidate_doc_filter(question, not self.runtime.a_board.use_doc_ids_as_hint_only)
        base_queries_by_option = build_option_queries(question)
        option_candidates = retrieve_option_candidates(
            self.retriever.index,
            question,
            filter_doc_ids=filter_doc_ids,
            top_k_per_query=self.retriever.top_k_per_query,
            fused_top_k=self.runtime.a_board.max_option_candidates,
        )
        verdicts: dict[str, OptionVerdict] = {}
        option_runs: dict[str, dict] = {}
        all_evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()
        threshold = self.runtime.a_board.low_confidence_threshold
        verdict_profile = self.llm.settings.thinking_profile_for_step("multi_logicrag_option_verdict")
        retry_profile = self.llm.settings.thinking_profile_for_step("multi_logicrag_option_retry")

        for option_key, option_text in sorted(question.options.items()):
            option_question = multi_logicrag.build_multi_option_question(question, option_key, option_text)
            candidates = option_candidates.get(option_key, [])
            evidence = self._option_compressor(question.domain).compress(option_question, candidates)
            coverage = assess_doc_coverage(question.doc_ids, evidence).to_dict() if question.doc_ids else {}
            sufficiency = analyze_evidence_sufficiency(
                build_retrieval_target(question, f"{option_key} {option_text}"),
                [item.evidence_text for item in evidence],
            )
            response = self.llm.chat(
                build_option_evidence_judgement_messages(question, option_key, option_text, evidence),
                temperature=0.0,
                thinking_profile=verdict_profile,
            )
            total_usage.add(response.usage)
            verdict = parse_option_verdict(response.text, option_key)
            retried = False
            retry_reason = ""
            retry_queries: list[str] = []
            retry_raw_response = ""
            refinement_triggered = False
            refinement_reason = ""
            refinement_goal = ""
            refined_queries: list[str] = []

            if self.runtime.a_board.multi_logicrag_retry_enabled and multi_logicrag.should_expand_uncertain_option(
                verdict,
                coverage=coverage,
                threshold=threshold,
            ):
                retried = True
                retry_reason = "uncertain_option"
                retry_queries = multi_logicrag.build_retry_queries(
                    question,
                    option_key,
                    option_text,
                    base_queries_by_option.get(option_key, []),
                )
                gap_query = multi_logicrag.build_gap_aware_retry_query(question, option_key, option_text, evidence)
                if gap_query and gap_query not in retry_queries:
                    retry_queries.append(gap_query)
                refinement_triggered, refinement_reason = should_trigger_retrieval_refinement(domain=question.domain, sufficiency=sufficiency)
                if refinement_triggered:
                    refinement = build_lightweight_refined_queries(
                        question_text=question.question,
                        option_key=option_key,
                        option_text=option_text,
                        sufficiency=sufficiency,
                        prior_queries=retry_queries,
                    )
                    refinement_goal = refinement.goal
                    refined_queries = list(refinement.refined_queries)
                    for query in refinement.refined_queries:
                        if query not in retry_queries:
                            retry_queries.append(query)
                ranked_lists = []
                per_query_top_k = self.retriever.top_k_per_query + max(2, self.runtime.a_board.max_verifier_candidates_per_option // 2)
                fused_top_k = self.runtime.a_board.max_option_candidates + self.runtime.a_board.max_verifier_candidates_per_option
                for query in retry_queries:
                    ranked_lists.append(
                        self.retriever.index.search(
                            query=query,
                            top_k=per_query_top_k,
                            filter_doc_ids=filter_doc_ids,
                            source=f"multi_logicrag_retry_{option_key}",
                        )
                    )
                expanded_candidates = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k) if ranked_lists else []
                candidates = multi_logicrag.merge_unique_evidence(candidates, expanded_candidates)
                evidence = self._option_compressor(question.domain).compress(option_question, candidates)
                coverage = assess_doc_coverage(question.doc_ids, evidence).to_dict() if question.doc_ids else {}
                sufficiency = analyze_evidence_sufficiency(
                    build_retrieval_target(question, f"{option_key} {option_text}"),
                    [item.evidence_text for item in evidence],
                )
                retry_response = self.llm.chat(
                    build_option_evidence_judgement_messages(question, option_key, option_text, evidence),
                    temperature=0.0,
                    thinking_profile=retry_profile,
                )
                total_usage.add(retry_response.usage)
                retry_raw_response = retry_response.text
                verdict = parse_option_verdict(retry_response.text, option_key)

            for item in evidence:
                if item.chunk_id not in seen_chunks:
                    all_evidence.append(item)
                    seen_chunks.add(item.chunk_id)
            verdicts[option_key] = verdict
            option_runs[option_key] = {
                "relation": _relation_from_option_verdict(verdict),
                "confidence": float(verdict.confidence or 0.0),
                "retried": retried,
                "retry_reason": retry_reason,
                "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in evidence)),
                "evidence_chunk_ids": [item.chunk_id for item in evidence],
                "coverage": coverage,
                "sufficiency": sufficiency,
                "refinement_triggered": refinement_triggered,
                "refinement_reason": refinement_reason,
                "refinement_goal": refinement_goal,
                "refined_queries": refined_queries,
                "retry_queries": retry_queries,
                "support_evidence": list(verdict.support_evidence),
                "refute_evidence": list(verdict.refute_evidence),
                "reason": verdict.reason,
                "raw_response": verdict.raw_response,
                "retry_raw_response": retry_raw_response,
            }

        answer = multi_logicrag.assemble_multi_logicrag_answer(verdicts)
        confidence = _average_option_verdict_confidence(verdicts)
        fallback_record: dict | None = None
        final_evidence = _limit_evidence_with_doc_coverage(all_evidence, self.compressor.top_k, question.doc_ids)
        final_evidence, coverage_report = self._apply_a_board_coverage_gate(question, all_evidence, final_evidence)
        if not answer:
            fallback_record = self._single_pass_multi_fallback(question, total_usage, all_evidence, seen_chunks)
            fallback_answer = str(fallback_record.get("answer", ""))
            if fallback_answer:
                answer = fallback_answer
                confidence = float(fallback_record.get("confidence", 0.0) or 0.0)
            else:
                answer = _fallback_multi_option([verdict.to_dict() for verdict in verdicts.values()]) or "A"
                confidence = 0.0

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=final_evidence,
            token_usage=total_usage,
            raw_response="",
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "multi_logicrag",
                "option_runs": option_runs,
                "option_verdicts": {key: verdict.to_dict() for key, verdict in verdicts.items()},
                "coverage_report": coverage_report,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
                "fallback": fallback_record,
            },
        )

    def _solve_by_option_matrix(self, question: Question) -> AnswerResult:
        """A 榜质量模式：逐选项检索 + support/refute/insufficient judgement。"""
        total_usage = TokenUsage()
        filter_doc_ids = self.retriever._candidate_doc_filter(question, not self.runtime.a_board.use_doc_ids_as_hint_only)
        option_candidates = retrieve_option_candidates(
            self.retriever.index,
            question,
            filter_doc_ids=filter_doc_ids,
            top_k_per_query=self.retriever.top_k_per_query,
            fused_top_k=self.runtime.a_board.max_option_candidates,
        )
        verdicts: dict[str, OptionVerdict] = {}
        option_coverage: dict[str, dict] = {}
        all_evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()

        for option_key, option_text in sorted(question.options.items()):
            candidates = option_candidates.get(option_key, [])
            option_coverage[option_key] = {
                "candidate_count": len(candidates),
                "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in candidates)),
                "missing": len(candidates) == 0,
            }
            option_question = Question(
                qid=f"{question.qid}:{option_key}",
                domain=question.domain,
                split=question.split,
                question=question.question,
                options={option_key: option_text},
                answer_format="tf",
                type=question.type,
                doc_ids=question.doc_ids,
            )
            evidence = self._option_compressor(question.domain).compress(option_question, candidates)
            for item in evidence:
                if item.chunk_id not in seen_chunks:
                    all_evidence.append(item)
                    seen_chunks.add(item.chunk_id)
            messages = build_option_evidence_judgement_messages(question, option_key, option_text, evidence)
            response = self.llm.chat(
                messages,
                temperature=0.0,
                thinking_profile=self.llm.settings.thinking_profile_for_step("option_judgement"),
            )
            total_usage.add(response.usage)
            verdicts[option_key] = parse_option_verdict(response.text, option_key)

        final_evidence = _limit_evidence_with_doc_coverage(all_evidence, self.compressor.top_k, question.doc_ids)
        final_evidence, coverage_report = self._apply_a_board_coverage_gate(question, all_evidence, final_evidence)
        answer = synthesize_answer(verdicts, question.answer_format)
        confidence = _average_option_verdict_confidence(verdicts)
        fallback_record: dict | None = None
        if not answer:
            fallback_record = self._single_pass_multi_fallback(question, total_usage, all_evidence, seen_chunks)
            fallback_answer = str(fallback_record.get("answer", ""))
            if fallback_answer:
                answer = fallback_answer
                confidence = float(fallback_record.get("confidence", 0.0) or 0.0)
            else:
                answer = _fallback_multi_option([verdict.to_dict() for verdict in verdicts.values()]) or "A"
                confidence = 0.0

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=final_evidence,
            token_usage=total_usage,
            raw_response="",
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "option_matrix",
                "option_verdicts": {key: verdict.to_dict() for key, verdict in verdicts.items()},
                "option_coverage": option_coverage,
                "coverage_report": coverage_report,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
                "fallback": fallback_record,
            },
        )

    def _solve_multi_by_option(self, question: Question) -> AnswerResult:
        """多选题逐选项判断，降低整体 Prompt 漏选/误选风险。"""
        accepted: list[str] = []
        option_records: list[dict] = []
        all_evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()
        total_usage = TokenUsage()

        option_items = list(sorted(question.options.items()))

        def judge_option(item: tuple[str, str]) -> dict:
            option_key, option_text = item
            option_question = Question(
                qid=f"{question.qid}:{option_key}",
                domain=question.domain,
                split=question.split,
                question=f"{question.question}\n判断选项{option_key}是否正确：{option_text}",
                options={option_key: option_text},
                answer_format="tf",
                type=question.type,
                doc_ids=question.doc_ids,
            )
            retrieved = self.retriever.retrieve(option_question)
            evidence = self._option_compressor(question.domain).compress(option_question, retrieved)
            option_profile = self.llm.settings.thinking_profile_for_step("option_judgement")
            messages = build_option_judgement_messages(question, option_key, option_text, evidence)
            response = self.llm.chat(
                messages,
                temperature=0.0,
                thinking_profile=option_profile,
            )
            return {
                "option": option_key,
                "verdict": parse_verdict(response.text),
                "confidence": _extract_confidence(response.text),
                "token_usage": response.usage,
                "raw_response": response.text,
                "reasoning": response.reasoning,
                "evidence": evidence,
            }

        judged = parallel_map_ordered(
            option_items,
            judge_option,
            max_workers=min(self.runtime.concurrency.qwen_workers, max(1, len(option_items))),
        )

        for record in judged:
            verdict = record["verdict"]
            confidence = record["confidence"]
            usage = record["token_usage"]
            evidence = record["evidence"]
            total_usage.add(usage)
            if verdict is True:
                accepted.append(record["option"])
            for item in evidence:
                if item.chunk_id not in seen_chunks:
                    all_evidence.append(item)
                    seen_chunks.add(item.chunk_id)
            option_records.append(
                {
                    "option": record["option"],
                    "verdict": verdict,
                    "confidence": confidence,
                    "token_usage": usage.to_dict(),
                    "raw_response": record["raw_response"],
                    "reasoning": record["reasoning"],
                    "evidence_doc_ids": [item.doc_id for item in evidence],
                }
            )

        answer = "".join(sorted(set(accepted)))
        confidence = _average_known_confidence(option_records)
        fallback_record: dict | None = None
        final_evidence = _limit_evidence_with_doc_coverage(all_evidence, self.compressor.top_k, question.doc_ids)
        final_evidence, coverage_report = self._apply_a_board_coverage_gate(question, all_evidence, final_evidence)
        if not answer:
            # 全部选项都判 false 通常意味着证据或逐项提示存在盲点，追加一次整体复核。
            fallback_record = self._single_pass_multi_fallback(question, total_usage, all_evidence, seen_chunks)
            fallback_answer = str(fallback_record.get("answer", ""))
            if fallback_answer:
                answer = fallback_answer
                confidence = float(fallback_record.get("confidence", 0.0) or 0.0)
            else:
                # 只有整体复核仍失败时才使用非法答案兜底，便于报告中识别低置信风险。
                answer = _fallback_multi_option(option_records) or "A"
                confidence = 0.0

        return AnswerResult(
            qid=question.qid,
            answer=answer,
            confidence=confidence,
            evidence=final_evidence,
            token_usage=total_usage,
            raw_response="",
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "multi_option_judgement",
                "option_judgements": option_records,
                "coverage_report": coverage_report,
                "domain_coverage_facets": expected_evidence_facets(question.domain, question.question) if coverage_report or self.runtime.a_board.coverage_gate_enabled else [],
                "fallback": fallback_record,
            },
        )

    def _single_pass_multi_fallback(
        self,
        question: Question,
        total_usage: TokenUsage,
        all_evidence: list[RetrievalResult],
        seen_chunks: set[str],
    ) -> dict:
        """逐选项全否时追加整体复核，并把 Token 与证据合并回主结果。"""
        fallback_profile = self.llm.settings.thinking_profile_for_step("multi_option_fallback")
        retrieved = self.retriever.retrieve(question)
        evidence = self.compressor.compress(question, retrieved)
        evidence, coverage_report = self._apply_a_board_coverage_gate(question, retrieved, evidence)
        messages = build_answer_messages(question, evidence)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            thinking_profile=fallback_profile,
        )
        total_usage.add(response.usage)
        for item in evidence:
            if item.chunk_id not in seen_chunks:
                all_evidence.append(item)
                seen_chunks.add(item.chunk_id)
        return {
            "strategy": "single_pass_after_all_false",
            "answer": parse_answer(response.text, question.answer_format),
            "confidence": _extract_confidence(response.text),
            "token_usage": response.usage.to_dict(),
            "raw_response": response.text,
            "reasoning": response.reasoning,
            "coverage_report": coverage_report,
            "evidence_doc_ids": [item.doc_id for item in evidence],
        }

    def _maybe_extract_financial_metrics(
        self,
        question: Question,
        evidence: list[RetrievalResult],
        total_usage: TokenUsage | None = None,
    ) -> dict | None:
        """财报题可选：抽取结构化指标 metadata，暂不改写最终答案。"""
        if question.domain != "financial_reports" or not self.runtime.a_board.financial_calculator_enabled:
            return None
        messages = build_financial_metric_extraction_messages(question, evidence)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            thinking_profile=self.llm.settings.thinking_profile_for_step("option_judgement"),
        )
        if total_usage is not None:
            total_usage.add(response.usage)
        return extract_json_object(response.text) or {"raw_response": response.text}

    def _apply_a_board_coverage_gate(
        self,
        question: Question,
        retrieved: list[RetrievalResult],
        evidence: list[RetrievalResult],
    ) -> tuple[list[RetrievalResult], dict]:
        """A 榜质量模式：若 evidence 缺少 doc_ids，则补检索缺失文档后重压缩。"""
        if not self.runtime.a_board.coverage_gate_enabled:
            return evidence, {}
        report = assess_doc_coverage(question.doc_ids, evidence)
        if report.ok or not report.missing_doc_ids:
            return evidence, report.to_dict()
        query = self.retriever._question_with_options(question)
        supplemental = retrieve_missing_doc_evidence(
            self.retriever.index,
            query=query,
            missing_doc_ids=report.missing_doc_ids,
            top_k=max(3, self.retriever.top_k_per_query // 2),
        )
        merged = [*retrieved, *supplemental]
        recompressed = self.compressor.compress(question, merged)
        final_report = assess_doc_coverage(question.doc_ids, recompressed)
        return recompressed, final_report.to_dict()

    def _option_compressor(self, domain: str) -> RuleEvidenceCompressor:
        """按领域选择逐选项证据预算，兼顾 Token 与证据充分性。"""
        if domain not in self.option_compressors:
            top_k = self.llm.settings.option_top_k_evidence
            max_chars = self.llm.settings.option_evidence_chars
            if domain == "research":
                top_k = max(top_k, 8)
                max_chars = max(max_chars, 6000)
            elif domain in {"financial_reports", "insurance"}:
                top_k = max(top_k, 6)
                max_chars = max(max_chars, 5000)
            elif domain in {"financial_contracts", "regulatory"}:
                top_k = min(top_k, 6)
                max_chars = min(max_chars, 5000)
            self.option_compressors[domain] = RuleEvidenceCompressor(max_chars=max_chars, top_k=top_k)
        return self.option_compressors[domain]


def _extract_confidence(text: str) -> float:
    """从模型 JSON 输出中抽取置信度；非法输出统一按低置信处理。"""
    try:
        obj = extract_json_object(text) or {}
        value = obj.get("confidence", None)
        if value is None:
            match = re.search(r'(?i)["\']?confidence["\']?\s*[:：]\s*([01](?:\.\d+)?)', text or "")
            value = match.group(1) if match else 0.0
        value = float(value)
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.0


def _average_known_confidence(records: list[dict]) -> float:
    """汇总逐选项置信度；无法解析的项按 0 处理。"""
    if not records:
        return 0.0
    values = [float(record.get("confidence", 0.0) or 0.0) for record in records]
    return sum(values) / len(values)



def _average_option_verdict_confidence(verdicts: dict[str, OptionVerdict]) -> float:
    """汇总 option-matrix verdict 置信度。"""
    if not verdicts:
        return 0.0
    values = [float(verdict.confidence or 0.0) for verdict in verdicts.values()]
    return sum(values) / len(values)



def _relation_from_option_verdict(verdict: OptionVerdict) -> str:
    if verdict.verdict is True:
        return "support"
    if verdict.verdict is False:
        return "refute"
    return "insufficient"



def _fallback_multi_option(records: list[dict]) -> str:
    """当所有选项均未判 true 时，选择置信度最高的不确定项兜底。"""
    unknown = [record for record in records if record.get("verdict") is None]
    if not unknown:
        return ""
    best = max(unknown, key=lambda record: float(record.get("confidence", 0.0) or 0.0))
    return str(best.get("option", ""))


def _limit_evidence_with_doc_coverage(
    evidence: list[RetrievalResult],
    top_k: int,
    doc_ids: list[str],
) -> list[RetrievalResult]:
    """截断聚合证据时继续保留多文档覆盖。"""
    selected: list[RetrievalResult] = []
    seen_chunks: set[str] = set()
    for doc_id in dict.fromkeys(doc_ids):
        for item in evidence:
            if item.doc_id == doc_id and item.chunk_id not in seen_chunks:
                selected.append(item)
                seen_chunks.add(item.chunk_id)
                break
        if len(selected) >= top_k:
            return selected
    for item in evidence:
        if item.chunk_id in seen_chunks:
            continue
        selected.append(item)
        seen_chunks.add(item.chunk_id)
        if len(selected) >= top_k:
            break
    return selected
