"""单题求解器：串联检索、压缩、Qwen 作答和答案解析。"""

from __future__ import annotations

import re

from agent.compress.rule_filter import RuleEvidenceCompressor
from agent.llm.qwen_client import QwenClient
from agent.reasoning import logicrag
from agent.reasoning.answer_parser import extract_json_object, parse_answer, parse_verdict
from agent.reasoning.prompts import (
    build_answer_messages,
    build_logicrag_final_compose_messages,
    build_option_judgement_messages,
)
from agent.retrieve.retriever import Retriever
from agent.schemas import AnswerResult, Question, RetrievalResult, TokenUsage


class Solver:
    """Agent 的最小闭环执行单元。"""

    def __init__(self, retriever: Retriever, compressor: RuleEvidenceCompressor, llm: QwenClient) -> None:
        self.retriever = retriever
        self.compressor = compressor
        self.llm = llm
        self.option_compressors: dict[str, RuleEvidenceCompressor] = {}

    def solve(self, question: Question) -> AnswerResult:
        """处理一道题并返回可提交答案与可审计证据。"""
        if self.llm.settings.retrieval_strategy == "logicrag_agent" and self.llm.settings.logicrag_enabled:
            return self._solve_logicrag_agent(question)
        if question.answer_format == "multi" and self.llm.settings.enable_multi_option_judgement:
            return self._solve_multi_by_option(question)

        retrieved = self.retriever.retrieve(question)
        evidence = self.compressor.compress(question, retrieved)
        messages = build_answer_messages(question, evidence)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            max_tokens=self.llm.settings.answer_max_tokens,
            enable_thinking=self.llm.settings.answer_enable_thinking,
        )
        answer = parse_answer(response.text, question.answer_format)
        confidence = _extract_confidence(response.text)
        if not answer:
            # 格式解析失败时给出可提交兜底值，同时把置信度降为 0，便于后续复核。
            answer = "A"
            confidence = 0.0
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

        rank_groups = logicrag.build_rankwise_query_groups(question, plan)
        filter_doc_ids = self.retriever._candidate_doc_filter(question, True)
        rank_memories: list[dict] = []
        combined: list[RetrievalResult] = []
        seen_chunks: set[str] = set()

        for group in rank_groups:
            rank_queries = logicrag.build_rankwise_queries_for_group(question, group, prior_memories=rank_memories)
            ranked_lists = [
                self.retriever.index.search(
                    query=query,
                    top_k=max(1, min(self.retriever.top_k_per_query, self.llm.settings.logicrag_rank_top_k)),
                    filter_doc_ids=filter_doc_ids,
                    source=f"logicrag_agent_rank_{group['rank']}",
                )
                for query in rank_queries
            ]
            rank_results = logicrag.reciprocal_rank_fusion(ranked_lists, top_k=self.llm.settings.logicrag_rank_top_k)
            for item in rank_results:
                if item.chunk_id not in seen_chunks:
                    combined.append(item)
                    seen_chunks.add(item.chunk_id)
            response = logicrag.summarize_rank_memory_with_qwen(
                question,
                self.llm,
                rank=group["rank"],
                nodes=group["nodes"],
                evidence=rank_results,
                prior_memories=rank_memories,
                max_chars=self.llm.settings.logicrag_memory_chars,
            )
            total_usage.add(response.usage)
            rank_memories.append(
                {
                    "rank": group["rank"],
                    "summary": response.text,
                    "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in rank_results))[:3],
                }
            )

        evidence = self.compressor.compress(question, combined)
        messages = build_logicrag_final_compose_messages(question, evidence, plan, rank_memories)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            max_tokens=self.llm.settings.answer_max_tokens,
            enable_thinking=self.llm.settings.answer_enable_thinking,
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
            evidence=evidence,
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
            },
        )

    def _solve_multi_by_option(self, question: Question) -> AnswerResult:
        """多选题逐选项判断，降低整体 Prompt 漏选/误选风险。"""
        accepted: list[str] = []
        option_records: list[dict] = []
        all_evidence: list[RetrievalResult] = []
        seen_chunks: set[str] = set()
        total_usage = TokenUsage()

        for option_key, option_text in sorted(question.options.items()):
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
            messages = build_option_judgement_messages(question, option_key, option_text, evidence)
            response = self.llm.chat(
                messages,
                temperature=0.0,
                max_tokens=self.llm.settings.option_judgement_max_tokens,
                enable_thinking=self.llm.settings.option_judgement_enable_thinking,
            )
            verdict = parse_verdict(response.text)
            confidence = _extract_confidence(response.text)
            total_usage.add(response.usage)
            if verdict is True:
                accepted.append(option_key)
            for item in evidence:
                if item.chunk_id not in seen_chunks:
                    all_evidence.append(item)
                    seen_chunks.add(item.chunk_id)
            option_records.append(
                {
                    "option": option_key,
                    "verdict": verdict,
                    "confidence": confidence,
                    "token_usage": response.usage.to_dict(),
                    "raw_response": response.text,
                    "reasoning": response.reasoning,
                    "evidence_doc_ids": [item.doc_id for item in evidence],
                }
            )

        answer = "".join(sorted(set(accepted)))
        confidence = _average_known_confidence(option_records)
        fallback_record: dict | None = None
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
            evidence=_limit_evidence_with_doc_coverage(all_evidence, self.compressor.top_k, question.doc_ids),
            token_usage=total_usage,
            raw_response="",
            metadata={
                "answer_format": question.answer_format,
                "domain": question.domain,
                "strategy": "multi_option_judgement",
                "option_judgements": option_records,
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
        retrieved = self.retriever.retrieve(question)
        evidence = self.compressor.compress(question, retrieved)
        messages = build_answer_messages(question, evidence)
        response = self.llm.chat(
            messages,
            temperature=0.0,
            max_tokens=self.llm.settings.answer_max_tokens,
            enable_thinking=self.llm.settings.answer_enable_thinking,
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
            "evidence_doc_ids": [item.doc_id for item in evidence],
        }

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
