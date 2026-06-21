"""LogicRAG 规划、rank-wise 检索/记忆与 retrieval-first 查询构造。"""

from __future__ import annotations

from agent.llm.qwen_client import LLMResponse, QwenClient
from agent.reasoning.answer_parser import extract_json_object
from agent.reasoning.prompts import (
    build_logicrag_memory_summary_messages,
    build_logicrag_plan_messages,
    build_logicrag_query_bundle_messages,
    build_logicrag_refinement_messages,
    build_logicrag_sufficiency_messages,
)
from agent.reasoning.retrieval_refiner import (
    LogicRAGSufficiencyJudgement,
    parse_logicrag_query_bundles,
    parse_logicrag_sufficiency_judgement,
)
from agent.retrieve.context import build_evidence_packs, select_results_from_packs
from agent.retrieve.expansion import build_short_hypothetical_query, build_sparse_feedback_query
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.rerank import rerank_retrieval_results
from agent.retrieve.targets import analyze_evidence_sufficiency, build_retrieval_target, merge_retrieval_targets, question_with_options
from agent.schemas import LogicNode, LogicPlan, Question


def _default_logic_planner(question: Question, max_subproblems: int, max_ranks: int):
    """安全默认 planner：纯本地启发式，不发起任何外部模型调用。"""
    return _fallback_logic_plan(question, max_subproblems=max_subproblems, max_ranks=max_ranks)


DEFAULT_LOGIC_PLANNER = _default_logic_planner

PAPER_FAITHFUL_PLANNER_CORE = [
    "llm_subproblem_decomposition",
    "logical_dependency_dag",
    "topological_rank_execution",
    "same_rank_merge_ready",
]


def parse_logic_plan(text: str, max_subproblems: int = 6, max_ranks: int = 4) -> LogicPlan:
    obj = extract_json_object(text) or {}
    raw_nodes = obj.get("subproblems") or obj.get("nodes") or []
    nodes = [LogicNode.from_dict(row) for row in raw_nodes[:max_subproblems]]
    plan = LogicPlan(nodes=nodes, rationale=str(obj.get("rationale", "")).strip())
    return sanitize_logic_plan(plan, max_subproblems=max_subproblems, max_ranks=max_ranks)



def build_logicrag_rrf_queries(
    question: Question,
    max_subproblems: int = 6,
    max_ranks: int = 4,
    planner=None,
    seed_results=None,
) -> list[str]:
    plan = plan_logic_subproblems(
        question,
        max_subproblems=max_subproblems,
        max_ranks=max_ranks,
        planner=planner,
    )
    targets = [
        build_retrieval_target(question, node.text, node_id=node.node_id, rank=node.rank, doc_scope=question.doc_ids)
        for node in plan.nodes
    ]
    merged = merge_retrieval_targets(question, targets, node_id="rrf", rank=0)
    queries = [question_with_options(question), *merged.query_variants]
    short_hypo = build_short_hypothetical_query(merged, max_terms=4)
    if short_hypo:
        queries.append(short_hypo)
    if seed_results:
        feedback_query = build_sparse_feedback_query(
            question,
            merged,
            seed_results,
            idf_lookup=None,
            max_terms=4,
        )
        if feedback_query:
            queries.append(feedback_query)
    return _dedupe_queries(queries)[: max(4, max_subproblems * 2 + 2)]



def plan_logic_subproblems(
    question: Question,
    max_subproblems: int = 6,
    max_ranks: int = 4,
    planner=None,
) -> LogicPlan:
    planner_fn = planner if planner is not None else DEFAULT_LOGIC_PLANNER
    if planner_fn is None:
        return _fallback_logic_plan(question, max_subproblems=max_subproblems, max_ranks=max_ranks)

    raw_plan = planner_fn(question, max_subproblems=max_subproblems, max_ranks=max_ranks)
    if isinstance(raw_plan, LogicPlan):
        plan = raw_plan
    elif isinstance(raw_plan, str):
        plan = parse_logic_plan(raw_plan, max_subproblems=max_subproblems, max_ranks=max_ranks)
    elif isinstance(raw_plan, dict):
        plan = sanitize_logic_plan(
            LogicPlan.from_dict(raw_plan),
            max_subproblems=max_subproblems,
            max_ranks=max_ranks,
        )
    else:
        raise TypeError("planner must return LogicPlan, dict, or JSON string")

    return sanitize_logic_plan(plan, max_subproblems=max_subproblems, max_ranks=max_ranks)



def plan_logic_subproblems_with_qwen(
    question: Question,
    llm: QwenClient,
    max_subproblems: int = 6,
    max_ranks: int = 4,
) -> tuple[LogicPlan, LLMResponse]:
    messages = build_logicrag_plan_messages(question, max_subproblems=max_subproblems, max_ranks=max_ranks)
    response = llm.chat(
        messages,
        temperature=0.0,
        thinking_profile=llm.settings.thinking_profile_for_step("logicrag_planner"),
    )
    try:
        plan = parse_logic_plan(response.text, max_subproblems=max_subproblems, max_ranks=max_ranks)
        if not plan.nodes:
            raise ValueError("empty plan")
    except Exception:
        plan = _fallback_logic_plan(question, max_subproblems=max_subproblems, max_ranks=max_ranks)
        plan.metadata["planner_fallback"] = True
        plan.metadata["planner_contract"] = _planner_contract_payload(
            plan.metadata,
            applied_extensions=["heuristic_fallback"],
        )
    return plan, response



def sanitize_logic_plan(
    plan: LogicPlan,
    max_subproblems: int = 6,
    max_ranks: int = 4,
) -> LogicPlan:
    kept_nodes: list[LogicNode] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    dropped_empty = False
    dropped_duplicate = False
    dropped_missing_dependencies = False

    for index, node in enumerate(plan.nodes, start=1):
        candidate = LogicNode(
            node_id=(node.node_id or f"n{index}").strip() or f"n{index}",
            text=(node.text or "").strip(),
            depends_on=[str(dep).strip() for dep in node.depends_on if str(dep).strip()],
            rank=int(getattr(node, "rank", 0) or 0),
            pruned=bool(getattr(node, "pruned", False)),
            metadata=dict(getattr(node, "metadata", {})),
        )
        if not candidate.text:
            dropped_empty = True
            continue
        normalized = candidate.normalized_text()
        if normalized in seen_texts:
            dropped_duplicate = True
            continue
        if candidate.node_id in seen_ids:
            candidate.node_id = _next_node_id(seen_ids, index)
        seen_ids.add(candidate.node_id)
        seen_texts.add(normalized)
        kept_nodes.append(candidate)
        if len(kept_nodes) >= max_subproblems:
            break

    kept_ids = {node.node_id for node in kept_nodes}
    for node in kept_nodes:
        deduped_deps: list[str] = []
        for dep in node.depends_on:
            if dep in kept_ids and dep != node.node_id and dep not in deduped_deps:
                deduped_deps.append(dep)
            else:
                dropped_missing_dependencies = True
        node.depends_on = deduped_deps

    broke_cycles = _break_cycles_in_place(kept_nodes)
    applied_extensions: list[str] = []
    if dropped_missing_dependencies:
        applied_extensions.append("drop_missing_dependencies")
    if broke_cycles:
        applied_extensions.append("break_cycles")
    if dropped_duplicate:
        applied_extensions.append("drop_duplicate_subproblems")
    if dropped_empty:
        applied_extensions.append("drop_empty_subproblems")

    clean = LogicPlan(
        nodes=kept_nodes,
        rationale=plan.rationale,
        metadata={**dict(plan.metadata), "planner_contract": _planner_contract_payload(plan.metadata, applied_extensions)},
    )
    levels = clean.topological_levels()
    if len(levels) <= max_ranks:
        return clean

    allowed_ids = {node_id for level in levels[:max_ranks] for node_id in level}
    trimmed_nodes = [node for node in clean.nodes if node.node_id in allowed_ids]
    for node in trimmed_nodes:
        if node.rank >= max_ranks:
            node.pruned = True
        node.depends_on = [dep for dep in node.depends_on if dep in allowed_ids]
    trimmed = LogicPlan(
        nodes=trimmed_nodes,
        rationale=clean.rationale,
        metadata={
            **dict(clean.metadata),
            "planner_contract": _planner_contract_payload(clean.metadata, ["trim_excess_ranks"]),
        },
    )
    trimmed.topological_levels()
    return trimmed


def append_unresolved_subproblem(
    plan: LogicPlan,
    text: str,
    depends_on: list[str] | None = None,
    append_after_rank: int | None = None,
    reason: str = "retrieval_insufficient",
    metadata: dict | None = None,
) -> LogicPlan:
    candidate_text = str(text or "").strip()
    if not candidate_text:
        raise ValueError("text must not be empty")

    base = sanitize_logic_plan(
        plan,
        max_subproblems=max(len(plan.nodes) + 1, 1),
        max_ranks=max(len(plan.nodes) + 1, 1),
    )
    levels = base.topological_levels()
    if append_after_rank is None:
        append_after_rank = len(levels) - 1
    anchor_ids = levels[append_after_rank] if 0 <= append_after_rank < len(levels) else []
    node_map = base.node_map()

    merged_dependencies: list[str] = []
    for dep in [*(depends_on or []), *anchor_ids]:
        dep_id = str(dep).strip()
        if dep_id and dep_id in node_map and dep_id not in merged_dependencies:
            merged_dependencies.append(dep_id)

    new_node_id = _next_node_id(set(node_map), len(base.nodes) + 1)
    new_node_metadata = dict(metadata or {})
    if reason:
        new_node_metadata.setdefault("augmentation_reason", reason)
    augmented = sanitize_logic_plan(
        LogicPlan(
            nodes=[*base.nodes, LogicNode(node_id=new_node_id, text=candidate_text, depends_on=merged_dependencies, metadata=new_node_metadata)],
            rationale=base.rationale,
            metadata=dict(base.metadata),
        ),
        max_subproblems=max(len(base.nodes) + 1, 1),
        max_ranks=max(len(levels) + 1, 1),
    )
    events = list(augmented.metadata.get("dynamic_augmentations", []))
    events.append(
        {
            "node_id": new_node_id,
            "text": candidate_text,
            "depends_on": list(merged_dependencies),
            "append_after_rank": append_after_rank,
            "trigger": reason,
        }
    )
    augmented.metadata["dynamic_augmentations"] = events
    return augmented


def build_rankwise_query_groups(question: Question, plan: LogicPlan) -> list[dict]:
    levels = plan.topological_levels()
    node_map = plan.node_map()
    groups: list[dict] = []
    for rank, node_ids in enumerate(levels):
        nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map]
        targets = [
            build_retrieval_target(question, node.text, node_id=node.node_id, rank=rank, doc_scope=question.doc_ids)
            for node in nodes
        ]
        target = merge_retrieval_targets(question, targets, node_id=f"rank_{rank}", rank=rank)
        queries = [_build_unified_rank_query(question, nodes)]
        groups.append({"rank": rank, "nodes": nodes, "targets": targets, "target": target, "queries": queries})
    return groups


def build_rankwise_queries_for_group(question: Question, group: dict, prior_memories: list[dict] | None = None, seed_results=None) -> list[str]:
    target = group.get("target") or build_retrieval_target(question, question.question)
    queries = _dedupe_queries(list(group.get("queries", [])))[:1]
    memory_anchor = _memory_anchor_text(prior_memories or [])
    if memory_anchor and queries:
        queries.append(f"{queries[0]} {memory_anchor}")
    return _dedupe_queries(queries)[:2]




def retrieve_rankwise_evidence(
    retriever,
    question: Question,
    plan: LogicPlan,
    per_query_top_k: int,
    fused_top_k: int,
) -> tuple[list[dict], list]:
    filter_doc_ids = None
    if hasattr(retriever, "_candidate_doc_filter"):
        filter_doc_ids = retriever._candidate_doc_filter(question, True)
    elif question.doc_ids:
        filter_doc_ids = set(question.doc_ids)

    rank_runs: list[dict] = []
    combined = []
    seen_chunks: set[str] = set()
    prior_memories: list[dict] = []
    for group in build_rankwise_query_groups(question, plan):
        seed_results = []
        rank_queries = build_rankwise_queries_for_group(question, group, prior_memories=prior_memories, seed_results=seed_results)
        ranked_lists = []
        for query in rank_queries:
            ranked_lists.append(
                retriever.index.search(
                    query=query,
                    top_k=per_query_top_k,
                    filter_doc_ids=filter_doc_ids,
                    source=f"logicrag_agent_rank_{group['rank']}",
                )
            )
        fused = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
        reranked = rerank_retrieval_results(question, group["target"], fused, top_k=fused_top_k)
        if hasattr(retriever.index, "get_chunk") and hasattr(retriever.index, "result_from_chunk"):
            packs = build_evidence_packs(
                retriever.index,
                question,
                reranked,
                max_packs=min(4, max(2, fused_top_k)),
                neighbor_window=1,
                max_chunks_per_pack=4,
                max_chars_per_pack=2200,
            )
            selected_results = select_results_from_packs(
                retriever.index,
                question,
                packs,
                top_k=fused_top_k,
                max_chars=6000,
            )
        else:
            packs = []
            selected_results = []
        final_results = selected_results or reranked[:fused_top_k]
        sufficiency = analyze_evidence_sufficiency(group["target"], [item.evidence_text for item in final_results])
        rank_runs.append({**group, "queries": rank_queries, "seed_results": seed_results, "results": final_results, "packs": packs, "sufficiency": sufficiency})
        prior_memories.append(
            {
                "rank": group["rank"],
                "summary": _build_rank_memory_summary(group, final_results),
            }
        )
        for item in final_results:
            if item.chunk_id not in seen_chunks:
                combined.append(item)
                seen_chunks.add(item.chunk_id)
    return rank_runs, combined


def generate_rank_query_bundles_with_qwen(
    question: Question,
    llm: QwenClient,
    group: dict,
    prior_memories: list[dict],
    max_bundles: int,
) -> tuple[list[str], list, LLMResponse]:
    messages = build_logicrag_query_bundle_messages(
        question,
        rank=group["rank"],
        nodes=group.get("nodes", []),
        prior_memories=prior_memories,
        max_bundles=max_bundles,
    )
    response = llm.chat(
        messages,
        temperature=0.0,
        thinking_profile=llm.settings.thinking_profile_for_step("logicrag_query_bundle"),
    )
    bundles = parse_logicrag_query_bundles(response.text, max_bundles=max_bundles)
    queries = [bundle.query for bundle in bundles]
    if not queries:
        queries = build_rankwise_queries_for_group(question, group, prior_memories=prior_memories)
    return _dedupe_queries(queries), bundles, response


def judge_rank_evidence_sufficiency_with_qwen(
    question: Question,
    llm: QwenClient,
    group: dict,
    evidence: list,
) -> tuple[LogicRAGSufficiencyJudgement, LLMResponse]:
    messages = build_logicrag_sufficiency_messages(
        question,
        rank=group["rank"],
        nodes=group.get("nodes", []),
        evidence=evidence,
    )
    response = llm.chat(
        messages,
        temperature=0.0,
        thinking_profile=llm.settings.thinking_profile_for_step("logicrag_sufficiency_gate"),
    )
    return parse_logicrag_sufficiency_judgement(response.text), response


def refine_rank_query_bundles_with_qwen(
    question: Question,
    llm: QwenClient,
    group: dict,
    evidence: list,
    judgement: LogicRAGSufficiencyJudgement,
    prior_queries: list[str],
    max_bundles: int,
) -> tuple[list[str], list, LLMResponse]:
    messages = build_logicrag_refinement_messages(
        question,
        rank=group["rank"],
        nodes=group.get("nodes", []),
        evidence=evidence,
        judgement=judgement,
        prior_queries=prior_queries,
        max_bundles=max_bundles,
    )
    response = llm.chat(
        messages,
        temperature=0.0,
        thinking_profile=llm.settings.thinking_profile_for_step("logicrag_refinement"),
    )
    bundles = parse_logicrag_query_bundles(response.text, max_bundles=max_bundles)
    return _dedupe_queries([bundle.query for bundle in bundles]), bundles, response


def retrieve_rankwise_evidence_adaptive(
    retriever,
    question: Question,
    plan: LogicPlan,
    llm: QwenClient,
    runtime,
    per_query_top_k: int,
    fused_top_k: int,
) -> tuple[list[dict], list, object]:
    from agent.schemas import TokenUsage

    total_usage = TokenUsage()
    base_filter_doc_ids, scope_policy = _initial_logicrag_filter_doc_ids(retriever, question, runtime)
    rank_runs: list[dict] = []
    combined = []
    seen_chunks: set[str] = set()
    prior_memories: list[dict] = []
    for group in build_rankwise_query_groups(question, plan):
        rank_results: list = []
        compose_results: list = []
        rounds: list[dict] = []
        current_filter_doc_ids = base_filter_doc_ids
        queries, bundles, response = generate_rank_query_bundles_with_qwen(
            question,
            llm,
            group,
            prior_memories=prior_memories,
            max_bundles=runtime.logicrag.max_query_bundles_per_rank,
        ) if runtime.logicrag.llm_query_bundles_enabled else (
            build_rankwise_queries_for_group(question, group, prior_memories=prior_memories),
            [],
            None,
        )
        if response is not None:
            total_usage.add(response.usage)
        exhausted = False
        final_sufficiency = {}
        max_rounds = max(0, int(runtime.logicrag.max_refinement_rounds_per_rank)) + 1
        for round_index in range(max_rounds):
            ranked_lists = [
                retriever.index.search(
                    query=query,
                    top_k=per_query_top_k,
                    filter_doc_ids=current_filter_doc_ids,
                    source=f"logicrag_agent_rank_{group['rank']}_round_{round_index}",
                )
                for query in queries
            ]
            fused = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
            reranked = rerank_retrieval_results(question, group["target"], fused, top_k=fused_top_k)
            final_results, packs = _select_logicrag_results(retriever, question, reranked, fused_top_k)
            compose_results = final_results
            for item in final_results:
                if item.chunk_id not in {x.chunk_id for x in rank_results}:
                    rank_results.append(item)
            judgement, suff_response = judge_rank_evidence_sufficiency_with_qwen(question, llm, group, final_results)
            total_usage.add(suff_response.usage)
            final_sufficiency = judgement.to_dict()
            rounds.append(
                {
                    "round": round_index,
                    "queries": list(queries),
                    "query_bundles": [bundle.to_dict() for bundle in bundles],
                    "filter_doc_ids": sorted(current_filter_doc_ids) if current_filter_doc_ids else None,
                    "evidence_doc_ids": list(dict.fromkeys(item.doc_id for item in final_results)),
                    "sufficiency": judgement.to_dict(),
                }
            )
            if judgement.sufficient:
                break
            if round_index >= max_rounds - 1:
                exhausted = True
                break
            if current_filter_doc_ids is None and runtime.logicrag.b_board_scope_narrowing_enabled:
                current_filter_doc_ids = _narrow_filter_doc_ids_from_results(final_results, runtime.logicrag.narrowed_doc_top_n)
            queries, bundles, refine_response = refine_rank_query_bundles_with_qwen(
                question,
                llm,
                group,
                final_results,
                judgement,
                prior_queries=queries,
                max_bundles=runtime.logicrag.max_query_bundles_per_rank,
            )
            total_usage.add(refine_response.usage)
            if not queries:
                exhausted = True
                break
        sufficiency = final_sufficiency or analyze_evidence_sufficiency(group["target"], [item.evidence_text for item in rank_results])
        rank_runs.append(
            {
                **group,
                "queries": list(dict.fromkeys(query for row in rounds for query in row.get("queries", []))),
                "seed_results": [],
                "results": rank_results,
                "compose_results": compose_results or rank_results,
                "packs": [],
                "sufficiency": sufficiency,
                "adaptive_retrieval": {
                    "enabled": True,
                    "scope_policy": scope_policy,
                    "rounds": rounds,
                    "exhausted": exhausted,
                },
            }
        )
        prior_memories.append({"rank": group["rank"], "summary": _build_rank_memory_summary(group, compose_results or rank_results)})
        for item in rank_results:
            if item.chunk_id not in seen_chunks:
                combined.append(item)
                seen_chunks.add(item.chunk_id)
    return rank_runs, combined, total_usage


def _select_logicrag_results(retriever, question: Question, reranked: list, fused_top_k: int) -> tuple[list, list]:
    if hasattr(retriever.index, "get_chunk") and hasattr(retriever.index, "result_from_chunk"):
        packs = build_evidence_packs(
            retriever.index,
            question,
            reranked,
            max_packs=min(4, max(2, fused_top_k)),
            neighbor_window=1,
            max_chunks_per_pack=4,
            max_chars_per_pack=2200,
        )
        selected_results = select_results_from_packs(
            retriever.index,
            question,
            packs,
            top_k=fused_top_k,
            max_chars=6000,
        )
    else:
        packs = []
        selected_results = []
    return selected_results or reranked[:fused_top_k], packs


def _initial_logicrag_filter_doc_ids(retriever, question: Question, runtime) -> tuple[set[str] | None, str]:
    if question.doc_ids and not runtime.a_board.use_doc_ids_as_hint_only:
        return set(question.doc_ids), "strict_doc_ids"
    if question.doc_ids and hasattr(retriever, "_candidate_doc_filter"):
        return retriever._candidate_doc_filter(question, False), "doc_ids_as_hint"
    return None, "global_then_narrow"


def _narrow_filter_doc_ids_from_results(results: list, top_n: int) -> set[str] | None:
    scores: dict[str, float] = {}
    for rank, item in enumerate(results):
        scores[item.doc_id] = scores.get(item.doc_id, 0.0) + max(0.0, float(item.score)) + 1.0 / (rank + 1)
    ranked = sorted(scores, key=scores.get, reverse=True)[: max(1, top_n)]
    return set(ranked) if ranked else None



def summarize_rank_memory_with_qwen(
    question: Question,
    llm: QwenClient,
    rank: int,
    nodes: list[LogicNode],
    evidence: list,
    prior_memories: list[dict],
    max_chars: int,
) -> LLMResponse:
    messages = build_logicrag_memory_summary_messages(
        question,
        rank=rank,
        nodes=nodes,
        evidence=evidence,
        prior_memories=prior_memories,
        max_chars=max_chars,
    )
    response = llm.chat(
        messages,
        temperature=0.0,
        thinking_profile=llm.settings.thinking_profile_for_step("logicrag_rank_summary"),
    )
    text = " ".join((response.text or "").split())
    if max_chars > 0:
        text = text[:max_chars]
    return LLMResponse(text=text or "无有效摘要。", usage=response.usage, reasoning=response.reasoning, raw=response.raw)



def _memory_anchor_text(prior_memories: list[dict]) -> str:
    if not prior_memories:
        return ""
    last_summary = str(prior_memories[-1].get("summary", "")).strip()
    if not last_summary:
        return ""
    clipped = " ".join(last_summary.split())[:80]
    return clipped



def _build_rank_memory_summary(group: dict, fused: list) -> str:
    node_text = " ".join(getattr(node, "text", "") for node in group.get("nodes", []))
    evidence_text = " ".join(getattr(item, "evidence_text", "") for item in fused[:2])
    summary = " ".join(part for part in [node_text, evidence_text] if part).strip()
    return " ".join(summary.split())[:80]


def _build_unified_rank_query(question: Question, nodes: list[LogicNode]) -> str:
    node_texts = _dedupe_queries(getattr(node, "text", "") for node in nodes)
    unified = " ".join(part for part in [question.question, *node_texts] if part).strip()
    return unified or question.question



def _break_cycles_in_place(nodes: list[LogicNode]) -> bool:
    node_map = {node.node_id: node for node in nodes}
    visited: set[str] = set()
    active: set[str] = set()
    changed = False

    def dfs(node_id: str) -> None:
        nonlocal changed
        visited.add(node_id)
        active.add(node_id)
        node = node_map[node_id]
        cleaned_deps: list[str] = []
        for dep in node.depends_on:
            if dep not in node_map:
                continue
            if dep == node_id:
                changed = True
                continue
            if dep in active:
                changed = True
                continue
            if dep not in visited:
                dfs(dep)
            cleaned_deps.append(dep)
        node.depends_on = cleaned_deps
        active.remove(node_id)

    for node in nodes:
        if node.node_id not in visited:
            dfs(node.node_id)
    return changed


def _planner_contract_payload(metadata: dict | None, applied_extensions: list[str]) -> dict:
    source = dict(metadata or {})
    existing = dict(source.get("planner_contract", {}))
    seen_extensions = list(existing.get("applied_extensions", []))
    for extension in applied_extensions:
        if extension not in seen_extensions:
            seen_extensions.append(extension)
    return {
        **existing,
        "paper_faithful_core": list(PAPER_FAITHFUL_PLANNER_CORE),
        "applied_extensions": seen_extensions,
    }


def _next_node_id(seen_ids: set[str], start: int) -> str:
    cursor = max(1, start)
    while True:
        candidate = f"n{cursor}"
        if candidate not in seen_ids:
            return candidate
        cursor += 1



def _fallback_logic_plan(question: Question, max_subproblems: int, max_ranks: int) -> LogicPlan:
    nodes = [LogicNode(node_id="n1", text=question.question, depends_on=[])]
    for index, (key, value) in enumerate(sorted(question.options.items()), start=2):
        nodes.append(LogicNode(node_id=f"n{index}", text=f"{key} {value}", depends_on=["n1"]))
    return sanitize_logic_plan(
        LogicPlan(nodes=nodes[:max_subproblems], rationale="fallback heuristic plan"),
        max_subproblems=max_subproblems,
        max_ranks=max_ranks,
    )



def _dedupe_queries(queries: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
