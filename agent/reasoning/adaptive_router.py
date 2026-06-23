"""V15 自适应策略路由 + LLM-as-a-Judge 效用重排。

借鉴 FinAgent-RAG (arXiv 2605.05409) 的自适应策略路由器，按题型/复杂度路由
thinking/no-thinking/PoT。借鉴 Two-Phase Retrieval (arXiv 2605.20684) 的
LLM-as-a-Judge 效用排序，用 Qwen 按"对回答此问题的有用性"重排 BM25 候选。

不引入 embedding，不是 embedding reranker，合规。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.llm.qwen_client import QwenClient
from agent.reasoning.answer_parser import extract_json_object
from agent.reasoning.pot_reasoner import needs_pot
from agent.schemas import Question, RetrievalResult, TokenUsage


@dataclass(frozen=True)
class RouterConfig:
    """策略路由配置。"""

    # 简单题（mcq/tf）用 no-thinking
    simple_max_tokens: int = 1024
    simple_enable_thinking: bool = False
    # 复杂题（multi）用 thinking
    complex_max_tokens: int = 1536
    complex_enable_thinking: bool = True
    # LLM Judge 配置
    judge_max_tokens: int = 256
    judge_top_k_input: int = 30  # 第一阶段 BM25 候选数
    judge_top_k_output: int = 5  # 第二阶段 Judge 后保留数
    judge_enable_thinking: bool = False


@dataclass
class RoutingDecision:
    """路由决策结果。"""

    strategy: str  # "simple_cot" | "complex_thinking" | "pot"
    enable_thinking: bool
    max_tokens: int
    reason: str


def route_question(question: Question, config: RouterConfig | None = None) -> RoutingDecision:
    """自适应策略路由。

    规则：
    1. 需要 PoT 数值推理 → pot
    2. multi 题型 → complex_thinking
    3. mcq/tf 题型 → simple_cot
    """
    cfg = config or RouterConfig()

    if needs_pot(question):
        return RoutingDecision(
            strategy="pot",
            enable_thinking=cfg.complex_enable_thinking,
            max_tokens=cfg.complex_max_tokens,
            reason="数值比较题，路由到 PoT + thinking",
        )

    if question.answer_format == "multi":
        return RoutingDecision(
            strategy="complex_thinking",
            enable_thinking=cfg.complex_enable_thinking,
            max_tokens=cfg.complex_max_tokens,
            reason="多选题，启用 thinking 深度推理",
        )

    return RoutingDecision(
        strategy="simple_cot",
        enable_thinking=cfg.simple_enable_thinking,
        max_tokens=cfg.simple_max_tokens,
        reason="单选/判断题，no-thinking 快速推理",
    )


def build_judge_messages(
    question: Question,
    candidates: list[RetrievalResult],
    config: RouterConfig | None = None,
) -> list[dict[str, str]]:
    """构建 LLM-as-a-Judge 效用排序 prompt。

    让 Qwen 按"对回答此问题的有用性"给每个候选打分。
    不是 embedding reranker，是 LLM 评分，合规。
    """
    cfg = config or RouterConfig()
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))

    # 截断候选文本避免 prompt 过长
    candidate_texts = []
    for i, c in enumerate(candidates[:cfg.judge_top_k_input]):
        text = (c.evidence_text or "")[:200]
        candidate_texts.append(f"[C{i+1}] doc={c.doc_id} page={c.metadata.get('page')}\n{text}")

    return [
        {
            "role": "system",
            "content": (
                "你是金融证据效用评估器。请评估每条候选证据对回答此问题的有用性。\n"
                "评分标准：\n"
                "  5=直接包含答案所需数值/事实\n"
                "  4=包含相关数值但需交叉引用\n"
                "  3=话题相关但不含关键数值\n"
                "  2=间接相关\n"
                "  1=不相关\n"
                "返回紧凑 JSON，不要 Markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"题干：{question.question}\n选项：\n{options}\n\n"
                f"候选证据：\n{chr(10).join(candidate_texts)}\n\n"
                '返回紧凑 JSON：{"scores":{"C1":5,"C2":3,"C3":4},'
                '"top_ids":["C1","C3","C5"]}'
            ),
        },
    ]


def llm_judge_rerank(
    question: Question,
    candidates: list[RetrievalResult],
    llm: QwenClient,
    config: RouterConfig | None = None,
) -> tuple[list[RetrievalResult], TokenUsage]:
    """LLM-as-a-Judge 效用重排。

    第一阶段：BM25 检索 Top-30（已有）
    第二阶段：Qwen 做 judge 按效用重新排序
    返回 Top-5 和 token usage。
    """
    cfg = config or RouterConfig()
    usage = TokenUsage()

    if len(candidates) <= cfg.judge_top_k_output:
        return candidates, usage

    # 截取 Top-30 候选
    pool = candidates[: cfg.judge_top_k_input]

    messages = build_judge_messages(question, pool, cfg)
    response = llm.chat(
        messages,
        temperature=0.0,
        max_tokens=cfg.judge_max_tokens,
        enable_thinking=cfg.judge_enable_thinking,
    )
    usage.add(response.usage)

    parsed = extract_json_object(response.text) or {}
    scores = parsed.get("scores", {})
    top_ids = parsed.get("top_ids", [])

    # 按 top_ids 顺序重排，如果 top_ids 为空则按 scores 降序
    if top_ids:
        id_to_candidate = {f"C{i+1}": c for i, c in enumerate(pool)}
        reranked = [id_to_candidate[cid] for cid in top_ids if cid in id_to_candidate]
        # 补充未被 top_ids 包含但分数高的
        used_indices = {int(cid[1:]) - 1 for cid in top_ids if cid.startswith("C") and cid[1:].isdigit()}
        for i, c in enumerate(pool):
            if i not in used_indices:
                reranked.append(c)
    elif scores:
        # 按 scores 降序排
        scored = [(scores.get(f"C{i+1}", 0), i, c) for i, c in enumerate(pool)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        reranked = [c for _, _, c in scored]
    else:
        # Judge 失败，保持原序
        reranked = list(pool)

    return reranked[: cfg.judge_top_k_output], usage
