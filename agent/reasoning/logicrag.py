"""LogicRAG 规划、rank-wise 检索/记忆与 retrieval-first 查询构造。"""

from __future__ import annotations

from agent.llm.qwen_client import LLMResponse, QwenClient
from agent.reasoning.answer_parser import extract_json_object
from agent.reasoning.prompts import build_logicrag_memory_summary_messages, build_logicrag_plan_messages
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.schemas import LogicNode, LogicPlan, Question


def _default_logic_planner(question: Question, max_subproblems: int, max_ranks: int):
    """安全默认 planner：纯本地启发式，不发起任何外部模型调用。"""
    return _fallback_logic_plan(question, max_subproblems=max_subproblems, max_ranks=max_ranks)


DEFAULT_LOGIC_PLANNER = _default_logic_planner


def parse_logic_plan(text: str, max_subproblems: int = 6, max_ranks: int = 4) -> LogicPlan:
    """从模型输出中解析 LogicPlan，并做最小稳定化处理。"""
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
) -> list[str]:
    """优先使用 planner 生成子问题；未接线时退化为本地启发式多查询。"""
    plan = plan_logic_subproblems(
        question,
        max_subproblems=max_subproblems,
        max_ranks=max_ranks,
        planner=planner,
    )
    queries = [question.question]
    option_text = " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))
    for node in plan.nodes:
        queries.append(f"{question.question} {node.text}")
        if option_text:
            queries.append(f"{node.text} {option_text}")
    return _dedupe_queries(queries)[: max(1, max_subproblems * 2 + 1)]


def plan_logic_subproblems(
    question: Question,
    max_subproblems: int = 6,
    max_ranks: int = 4,
    planner=None,
) -> LogicPlan:
    """把 planner 输出统一收敛为清洗后的 LogicPlan，便于测试时 monkeypatch。"""
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
    """显式调用 Qwen planner，并把 usage 返回给 solver 累计。"""
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
    return plan, response


def sanitize_logic_plan(
    plan: LogicPlan,
    max_subproblems: int = 6,
    max_ranks: int = 4,
) -> LogicPlan:
    """去空、去重、去非法依赖、断环，并按层级裁剪。"""
    kept_nodes: list[LogicNode] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

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
            continue
        normalized = candidate.normalized_text()
        if normalized in seen_texts:
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
        node.depends_on = deduped_deps

    _break_cycles_in_place(kept_nodes)

    clean = LogicPlan(nodes=kept_nodes, rationale=plan.rationale, metadata=dict(plan.metadata))
    levels = clean.topological_levels()
    if len(levels) <= max_ranks:
        return clean

    allowed_ids = {node_id for level in levels[:max_ranks] for node_id in level}
    trimmed_nodes = [node for node in clean.nodes if node.node_id in allowed_ids]
    for node in trimmed_nodes:
        if node.rank >= max_ranks:
            node.pruned = True
        node.depends_on = [dep for dep in node.depends_on if dep in allowed_ids]
    trimmed = LogicPlan(nodes=trimmed_nodes, rationale=clean.rationale, metadata=clean.metadata)
    trimmed.topological_levels()
    return trimmed


def build_rankwise_query_groups(question: Question, plan: LogicPlan) -> list[dict]:
    """按 DAG rank 组织子问题与检索查询。"""
    levels = plan.topological_levels()
    node_map = plan.node_map()
    option_text = " ".join(f"{key} {value}" for key, value in sorted(question.options.items()))
    groups: list[dict] = []
    for rank, node_ids in enumerate(levels):
        nodes = [node_map[node_id] for node_id in node_ids if node_id in node_map]
        queries = [question.question]
        for node in nodes:
            queries.append(f"{question.question} {node.text}")
            if option_text:
                queries.append(f"{node.text} {option_text}")
        groups.append({"rank": rank, "nodes": nodes, "queries": _dedupe_queries(queries)})
    return groups


def build_rankwise_queries_for_group(question: Question, group: dict, prior_memories: list[dict] | None = None) -> list[str]:
    """把 rank 级子问题展开为可检索查询；后续 rank 只追加 1 条 memory-aware 查询。"""
    queries = _dedupe_queries(list(group.get("queries", [])))[:2]
    memory_anchor = _memory_anchor_text(prior_memories or [])
    if memory_anchor and queries:
        queries.append(f"{queries[0]} {memory_anchor}")
    return _dedupe_queries(queries)[:3]


def retrieve_rankwise_evidence(
    retriever,
    question: Question,
    plan: LogicPlan,
    per_query_top_k: int,
    fused_top_k: int,
) -> tuple[list[dict], list]:
    """按 rank 发起 RRF 检索，返回每层结果及全局去重结果。"""
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
        rank_queries = build_rankwise_queries_for_group(question, group, prior_memories=prior_memories)
        ranked_lists = [
            retriever.index.search(
                query=query,
                top_k=per_query_top_k,
                filter_doc_ids=filter_doc_ids,
                source=f"logicrag_agent_rank_{group['rank']}",
            )
            for query in rank_queries
        ]
        fused = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
        rank_runs.append({**group, "queries": rank_queries, "results": fused})
        prior_memories.append(
            {
                "rank": group["rank"],
                "summary": _build_rank_memory_summary(group, fused),
            }
        )
        for item in fused:
            if item.chunk_id not in seen_chunks:
                combined.append(item)
                seen_chunks.add(item.chunk_id)
    return rank_runs, combined


def summarize_rank_memory_with_qwen(
    question: Question,
    llm: QwenClient,
    rank: int,
    nodes: list[LogicNode],
    evidence: list,
    prior_memories: list[dict],
    max_chars: int,
) -> LLMResponse:
    """显式调用 Qwen 做 rank-wise memory summary，并保留 usage。"""
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
    """从上游记忆中提取一个短锚点，供后续 rank 查询收缩范围。"""
    if not prior_memories:
        return ""
    last_summary = str(prior_memories[-1].get("summary", "")).strip()
    if not last_summary:
        return ""
    clipped = " ".join(last_summary.split())[:80]
    return clipped


def _build_rank_memory_summary(group: dict, fused: list) -> str:
    """为 retrieval-only helper 生成轻量 rank memory，避免空锚点传播。"""
    node_text = " ".join(getattr(node, "text", "") for node in group.get("nodes", []))
    evidence_text = " ".join(getattr(item, "evidence_text", "") for item in fused[:2])
    summary = " ".join(part for part in [node_text, evidence_text] if part).strip()
    return " ".join(summary.split())[:80]


def _break_cycles_in_place(nodes: list[LogicNode]) -> None:
    node_map = {node.node_id: node for node in nodes}
    visited: set[str] = set()
    active: set[str] = set()

    def dfs(node_id: str) -> None:
        visited.add(node_id)
        active.add(node_id)
        node = node_map[node_id]
        cleaned_deps: list[str] = []
        for dep in node.depends_on:
            if dep not in node_map or dep == node_id:
                continue
            if dep in active:
                continue
            if dep not in visited:
                dfs(dep)
            cleaned_deps.append(dep)
        node.depends_on = cleaned_deps
        active.remove(node_id)

    for node in nodes:
        if node.node_id not in visited:
            dfs(node.node_id)


def _next_node_id(seen_ids: set[str], start: int) -> str:
    cursor = max(1, start)
    while True:
        candidate = f"n{cursor}"
        if candidate not in seen_ids:
            return candidate
        cursor += 1


def _fallback_logic_plan(question: Question, max_subproblems: int, max_ranks: int) -> LogicPlan:
    """未接入真实 Qwen planner 时的本地兜底，保持 retrieval variant 可运行。"""
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
