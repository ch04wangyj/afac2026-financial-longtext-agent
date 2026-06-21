"""Qwen 作答与 LogicRAG Agent Prompt 模板。"""

from __future__ import annotations

import json

from agent.reasoning.fact_ledger import format_numeric_fact_ledger
from agent.schemas import Question, RetrievalResult


DOMAIN_ROLES = {
    "insurance": "你是保险精算与条款核验专家。",
    "regulatory": "你是金融合规律师，必须严格依据法条原文判断。",
    "financial_contracts": "你是固定收益分析师，必须严格依据债券募集说明书。",
    "financial_reports": "你是注册会计师，必须精确引用财报数值并注意单位。",
    "research": "你是金融行业研究员，必须依据研报证据判断。",
}


def build_answer_messages(question: Question, evidence: list[RetrievalResult]) -> list[dict[str, str]]:
    """构造单题消息，要求模型只依据证据并输出可解析 JSON。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    evidence_text = format_evidence(evidence)
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    user = f"""请只依据给定证据回答题目。

题型: {question.answer_format}
题目: {question.question}
选项:
{options}

证据:
{evidence_text}

输出 JSON，字段包括:
{{"answer":"A或ABCD","confidence":0到1之间的小数,"reason":"一句话依据"}}

规则:
- mcq/tf 只能输出单个大写字母。
- multi 输出所有正确选项，按字母排序，无分隔符。
- answer 不得为空；证据不足时选择最可能答案并降低 confidence。
- 证据不足时仅可降低 confidence；若必须作答，也只能给出受限猜测。
- 不要为了弥补检索不足而展开泛化推理。
- 不要输出冗长背景解释来补偿证据缺口。
"""
    return [
        {"role": "system", "content": f"{role} 不得使用外部知识，不得编造证据。"},
        {"role": "user", "content": user},
    ]


def build_option_judgement_messages(
    question: Question,
    option_key: str,
    option_text: str,
    evidence: list[RetrievalResult],
) -> list[dict[str, str]]:
    """构造逐选项判断 Prompt，输出 true/false 便于多选题聚合。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    evidence_text = format_evidence(evidence)
    user = f"""请只依据给定证据判断一个候选选项是否正确。

原题: {question.question}
待判断选项: {option_key}. {option_text}

证据:
{evidence_text}

输出 JSON，字段包括:
{{"verdict":true或false,"confidence":0到1之间的小数,"reason":"一句话依据"}}

规则:
- verdict=true 表示该选项准确，应进入多选答案。
- verdict=false 表示该选项不准确或证据不支持。
- 不得使用外部知识；证据不足时 verdict=false，并降低 confidence。
- 只输出一行 JSON，不要 Markdown，不要自我反思。
- reason 不超过 40 个中文字符。
"""
    return [
        {"role": "system", "content": f"{role} 不得使用外部知识，不得编造证据。"},
        {"role": "user", "content": user},
    ]


def build_option_evidence_judgement_messages(
    question: Question,
    option_key: str,
    option_text: str,
    evidence: list[RetrievalResult],
) -> list[dict[str, str]]:
    """构造逐选项证据支持/反驳/不足判断 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    user = f"""请只依据给定证据判断选项是否被支持、被反驳或证据不足。

题型: {question.answer_format}
原题: {question.question}
待判断选项: {option_key}. {option_text}

证据:
{format_evidence(evidence)}

只输出 JSON:
{{
  "option": "{option_key}",
  "relation": "support|refute|insufficient",
  "confidence": 0到1之间的小数,
  "support_evidence": ["证据编号，如[1]"],
  "refute_evidence": ["证据编号，如[2]"],
  "reason": "不超过60字"
}}

规则:
- support 表示该选项准确，应进入答案。
- refute 表示证据明确说明该选项错误。
- insufficient 表示证据不足，不能用常识猜测。
- 判断的是“该选项是否应被原题选中”，不能仅因括号内描述本身属实就判 support。
- 不得使用外部知识，不得编造证据编号。
- reason 必须引用关键原文事实，不要输出 Markdown。
"""
    return [
        {"role": "system", "content": f"{role} 你负责逐选项证据判断，只能依据给定证据。"},
        {"role": "user", "content": user},
    ]


def build_claim_set_verification_messages(
    question: Question,
    verdicts: dict[str, dict],
    claim_runs: dict[str, dict],
    evidence: list[RetrievalResult],
    fact_ledger: dict | None = None,
) -> list[dict[str, str]]:
    """构造逐选项完成后的集合级 exact-match 复核 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    evidence_positions = {item.chunk_id: index for index, item in enumerate(evidence, start=1)}
    local = {
        key: {
            "relation": value.get("relation"),
            "confidence": value.get("confidence"),
            # 局部判断和集合复核的证据编号空间不同，必须用 chunk_id 重映射。
            "support_evidence": [
                f"[{evidence_positions[chunk_id]}]"
                for chunk_id in value.get("support_chunk_ids", [])
                if chunk_id in evidence_positions
            ],
            "refute_evidence": [
                f"[{evidence_positions[chunk_id]}]"
                for chunk_id in value.get("refute_chunk_ids", [])
                if chunk_id in evidence_positions
            ],
            "support_chunk_ids": value.get("support_chunk_ids", []),
            "refute_chunk_ids": value.get("refute_chunk_ids", []),
            "calibration_tags": value.get("calibration_tags", []),
            "missing_slots": value.get("missing_slots", []),
            "missing_universal_doc_ids": value.get("missing_universal_doc_ids", []),
            "sufficiency": (claim_runs.get(key) or {}).get("sufficiency", {}),
        }
        for key, value in sorted(verdicts.items())
    }
    user = f"""请对逐选项结果做一次集合级复核，并输出最终精确答案。

题型: {question.answer_format}
题目: {question.question}
选项:
{options}

局部 verdict（只是候选，不是最终真值）:
{json.dumps(local, ensure_ascii=False)}

原始证据:
{format_evidence(evidence)}

数值事实账本:
{format_numeric_fact_ledger(fact_ledger or {})}

只输出 JSON:
{{
  "answer": "A或ABCD",
  "confidence": 0到1之间的小数,
  "option_relations": {{"A":"support|refute|insufficient"}},
  "reason": "不超过100字"
}}

复核规则:
- support 必须由有效证据编号支持；calibration_tags 或 missing_slots 不得被忽略。
- 含“均、都、双方、两份、两家、分别、所有”的选项，涉及的每份文档都必须有直接支持证据。
- 含“且、并且、同时、以及、；”的复合选项，每个子条件都成立才可 support；任一子条件错误即 refute。
- 数值、比例、同比、阈值和趋势只能依据数值事实账本及原始证据计算；必须统一单位，不得心算补值。
- multi 必须输出所有且仅有正确选项，按字母排序；mcq/tf 只能输出一个字母。
- answer 不得为空；即使证据仍不充分，也要给出最可能的合法选项集合并降低 confidence。
- 若局部 verdict 冲突，以可直接定位的原始证据和账本为准，不以局部 confidence 投票。
- 不得使用外部知识，不得编造证据或缺失数值。
"""
    return [
        {"role": "system", "content": f"{role} 你负责证据集合校准和 exact-match 最终复核。"},
        {"role": "user", "content": user},
    ]



def build_financial_metric_extraction_messages(
    question: Question,
    evidence: list[RetrievalResult],
) -> list[dict[str, str]]:
    """构造财报题结构化指标抽取 Prompt。"""
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items()))
    user = f"""请从证据中抽取财务指标原始数值。不要计算最终答案。

题目: {question.question}
选项:
{options}

证据:
{format_evidence(evidence)}

只输出 JSON:
{{
  "metric_values": [
    {{
      "entity": "公司或报告主体",
      "year": "年份",
      "metric": "指标名称",
      "value": "原文数值",
      "unit": "元|千元|万元|亿元|%|其他",
      "evidence_id": "[1]"
    }}
  ],
  "missing_metrics": ["缺失的指标"]
}}

规则:
- 只抽取证据中明确出现的数值。
- 保留原文单位，不要擅自换算。
- 不要计算最终答案，不要判断选项对错。
"""
    return [
        {"role": "system", "content": "你是财报数值抽取器，只抽取证据中的原始指标和单位。"},
        {"role": "user", "content": user},
    ]



def build_logicrag_plan_messages(
    question: Question,
    max_subproblems: int,
    max_ranks: int,
) -> list[dict[str, str]]:

    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items())) or "无"
    user = f"""你现在是 LogicRAG规划器。请把题目拆成可检索、可组合的子问题 DAG。

题型: {question.answer_format}
题目: {question.question}
选项:
{options}

约束:
- 最多输出 {max_subproblems} 个子问题。
- 最多 {max_ranks} 个依赖层级。
- 每个子问题必须能直接用于检索证据。
- 每个子问题必须指向可直接检索的具体事实、条款、数值或定义。
- 优先把子问题写成可直接检索的目标事实、目标数值、目标日期、目标条件或目标条款。
- 如果原题本质上是在比较、判断阈值、核对日期、核对条款或核对定义，子问题应先分别定位这些可检索对象，而不是只把原题换一种说法。
- depends_on 表示必须先解决的逻辑前置子问题，只能引用前面节点 id。
- 同一依赖层级的子问题后续会合并为一次检索；只有真正可以并行共享证据的子问题才应放在同一层。
- 不要输出空节点、重复节点或无意义改写。
- 不要只把原题换一种说法；要让每个子问题都对应一个更容易命中证据的检索目标。
- 不要把背景介绍、概念解释或大范围综述写成子问题。

只输出 JSON:
{{
  "subproblems": [
    {{"id": "n1", "text": "...", "depends_on": []}},
    {{"id": "n2", "text": "...", "depends_on": ["n1"]}}
  ],
  "rationale": "一句话说明拆分逻辑"
}}
"""
    return [
        {"role": "system", "content": f"{role} 你负责 LogicRAG 规划，不得编造法规或事实。"},
        {"role": "user", "content": user},
    ]


def build_logicrag_memory_summary_messages(
    question: Question,
    rank: int,
    nodes: list,
    evidence: list[RetrievalResult],
    prior_memories: list[dict],
    max_chars: int,
) -> list[dict[str, str]]:
    """构造按 rank 汇总局部记忆的 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    node_text = "\n".join(f"- {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}" for node in nodes) or "- 无"
    prior_text = "\n".join(
        f"rank={item.get('rank')}: {str(item.get('summary', ''))[:max_chars]}" for item in prior_memories
    ) or "无上游记忆"
    user = f"""你现在执行 LogicRAG memory summary。请汇总当前层级的证据，形成后续可复用短记忆。

rank={rank}
原题: {question.question}
当前层子问题:
{node_text}

已有上游记忆:
{prior_text}

当前证据:
{format_evidence(evidence)}

输出要求:
- 只输出纯文本摘要，不要 Markdown，不要 JSON。
- 摘要长度不超过 {max_chars} 字。
- 明确写出已确认事实、条件限制、仍不确定点。
- 仅依据证据和上游记忆，不得补充外部知识。
"""
    return [
        {"role": "system", "content": f"{role} 你负责 LogicRAG 分层记忆压缩，不得编造证据。"},
        {"role": "user", "content": user},
    ]


def build_logicrag_query_bundle_messages(
    question: Question,
    rank: int,
    nodes: list,
    prior_memories: list[dict],
    max_bundles: int,
) -> list[dict[str, str]]:
    """构造 LogicRAG rank 检索组合生成 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    node_text = "\n".join(f"- {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}" for node in nodes) or "- 无"
    memory_text = "\n".join(f"rank={item.get('rank')}: {item.get('summary', '')}" for item in prior_memories) or "无"
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items())) or "无"
    doc_scope = ", ".join(question.doc_ids) or "未限定"
    user = f"""你负责为 LogicRAG 第 {rank} 层直接提出检索组合。系统只做去重、格式规范化和执行检索，不会再替你做额外关键词工程。

原题: {question.question}
题型: {question.answer_format}
选项:
{options}
当前 doc scope: {doc_scope}
当前层子问题:
{node_text}
上游记忆:
{memory_text}

请输出 3 到 5 组检索组合，但最多 {max_bundles} 组。每组 query 应指向不同证据方向，例如主事实、实体强化、数值/日期、对比端点、条款后果。

只输出 JSON:
{{
  "query_bundles": [
    {{"query": "...", "intent": "...", "evidence_type": "metric_value|date|clause_consequence|definition|entity_fact|comparison_endpoint|other", "must_terms": ["..."], "doc_scope_hint": ["..."]}}
  ]
}}

规则:
- 不要依赖外部知识，不要编造事实。
- query 必须短而可检索，不要写成长推理。
- 不要只改写原题；要改变证据方向。
- 如果需要比较，至少覆盖比较双方或缺失端点。
- 如果是条款题，优先查找后果、处罚、期限、义务等可裁决条款。
"""
    return [
        {"role": "system", "content": f"{role} 你只负责生成可执行检索组合。"},
        {"role": "user", "content": user},
    ]


def build_logicrag_sufficiency_messages(
    question: Question,
    rank: int,
    nodes: list,
    evidence: list[RetrievalResult],
) -> list[dict[str, str]]:
    """构造 LogicRAG rank 证据充分性轻量判断 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    node_text = "\n".join(f"- {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}" for node in nodes) or "- 无"
    user = f"""你负责判断当前检索证据是否足够回答 LogicRAG 第 {rank} 层子问题。

原题: {question.question}
当前层子问题:
{node_text}

证据:
{format_evidence(evidence)}

只输出 JSON:
{{"sufficient": 0或1, "failure_tags": ["missing_metric_value_pair|missing_second_endpoint|same_doc_wrong_clause|missing_clause_consequence|generic_context_only|other"], "reason": "一句话", "missing_evidence": "缺什么证据", "next_search_goal": "下一轮应该搜什么"}}

规则:
- sufficient=1 表示证据足够支撑当前层进入记忆/最终组合。
- sufficient=0 表示信息不足；必须说明缺失证据类型和下一轮检索方向。
- 不要使用外部知识，不要替最终答案做无证据猜测。
"""
    return [
        {"role": "system", "content": f"{role} 你只做证据充分性判断，不做最终答题。"},
        {"role": "user", "content": user},
    ]


def build_logicrag_refinement_messages(
    question: Question,
    rank: int,
    nodes: list,
    evidence: list[RetrievalResult],
    judgement,
    prior_queries: list[str],
    max_bundles: int,
) -> list[dict[str, str]]:
    """构造 LogicRAG 检索方向重写 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    node_text = "\n".join(f"- {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}" for node in nodes) or "- 无"
    prior_query_text = "\n".join(f"- {query}" for query in prior_queries) or "- 无"
    reason = getattr(judgement, "reason", "") if not isinstance(judgement, dict) else judgement.get("reason", "")
    missing = getattr(judgement, "missing_evidence", "") if not isinstance(judgement, dict) else judgement.get("missing_evidence", "")
    next_goal = getattr(judgement, "next_search_goal", "") if not isinstance(judgement, dict) else judgement.get("next_search_goal", "")
    user = f"""上一轮 LogicRAG 第 {rank} 层检索证据不足。请重新思考下一轮检索方向，而不是只给上一轮 query 增加同义词。

原题: {question.question}
当前层子问题:
{node_text}
上一轮 query:
{prior_query_text}
不足原因: {reason}
缺失证据: {missing}
建议目标: {next_goal}

上一轮证据:
{format_evidence(evidence)}

请围绕缺失证据类型重定向，输出 3 到 5 组新的检索组合，最多 {max_bundles} 组。
只输出 JSON，格式同 query_bundles:
{{"query_bundles": [{{"query": "...", "intent": "...", "evidence_type": "...", "must_terms": ["..."], "doc_scope_hint": ["..."]}}]}}
"""
    return [
        {"role": "system", "content": f"{role} 你负责低可信检索方向重写，不得编造证据。"},
        {"role": "user", "content": user},
    ]


def build_logicrag_final_compose_messages(
    question: Question,
    evidence: list[RetrievalResult],
    logic_plan,
    rank_memories: list[dict],
) -> list[dict[str, str]]:

    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items())) or "无"
    plan_text = "\n".join(
        f"- rank={getattr(node, 'rank', 0)} {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}"
        for node in getattr(logic_plan, "nodes", [])
    ) or "无"
    memory_text = "\n".join(f"rank={item.get('rank')}: {item.get('summary', '')}" for item in rank_memories) or "无"
    user = f"""你现在执行 LogicRAG final compose。请以分层记忆作为主线，并仅把最终层证据当作收尾核验来作答。

题型: {question.answer_format}
题目: {question.question}
选项:
{options}

LogicRAG 规划:
{plan_text}

分层记忆:
{memory_text}

最终证据:
{format_evidence(evidence)}

输出 JSON，字段包括:
{{"answer":"A或ABCD","confidence":0到1之间的小数,"reason":"一句话依据"}}

规则:
- mcq/tf 只能输出单个大写字母。
- multi 输出所有正确选项，按字母排序，无分隔符。
- 严格依据分层记忆与最终层证据，不得使用外部知识。
- 分层记忆优先于最终层原文堆砌。
- 不要为了补偿上游检索缺口而扩写背景或常识推理。
- 若分层记忆与最终证据仍不足，只能降低 confidence。
"""
    return [
        {"role": "system", "content": f"{role} 你负责 LogicRAG 最终组合回答，不得编造证据。"},
        {"role": "user", "content": user},
    ]


def format_evidence(evidence: list[RetrievalResult]) -> str:
    """把检索证据格式化为 Prompt 中的可审计片段。"""
    if not evidence:
        return "未检索到证据。"
    lines: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        page = item.metadata.get("page")
        clause = item.metadata.get("clause_id", "")
        title = item.metadata.get("title", "")
        prefix = f"[{idx}] doc={item.doc_id} chunk={item.chunk_id} title={title} page={page} clause={clause}"
        lines.append(f"{prefix}\n{item.evidence_text}")
    return "\n\n".join(lines)
