"""Qwen 作答与 LogicRAG Agent Prompt 模板。"""

from __future__ import annotations

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
- 证据不足时仍需给出最可能答案，并降低 confidence。
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
- 不得使用外部知识，不得编造证据编号。
- reason 必须引用关键原文事实，不要输出 Markdown。
"""
    return [
        {"role": "system", "content": f"{role} 你负责逐选项证据判断，只能依据给定证据。"},
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
- depends_on 只能引用前面节点 id。
- 不要输出空节点、重复节点或无意义改写。

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


def build_logicrag_final_compose_messages(
    question: Question,
    evidence: list[RetrievalResult],
    logic_plan,
    rank_memories: list[dict],
) -> list[dict[str, str]]:
    """构造 LogicRAG 最终作答 Prompt。"""
    role = DOMAIN_ROLES.get(question.domain, "你是金融长文本问答专家。")
    options = "\n".join(f"{key}. {value}" for key, value in sorted(question.options.items())) or "无"
    plan_text = "\n".join(
        f"- rank={getattr(node, 'rank', 0)} {getattr(node, 'node_id', '')}: {getattr(node, 'text', '')}"
        for node in getattr(logic_plan, "nodes", [])
    ) or "无"
    memory_text = "\n".join(f"rank={item.get('rank')}: {item.get('summary', '')}" for item in rank_memories) or "无"
    user = f"""你现在执行 LogicRAG final compose。请综合规划、分层记忆与原始证据作答。

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
- 严格依据证据与分层记忆，不得使用外部知识。
- 证据不足时仍需给出最可能答案，并降低 confidence。
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
        prefix = f"[{idx}] doc={item.doc_id} title={title} page={page} clause={clause}"
        lines.append(f"{prefix}\n{item.evidence_text}")
    return "\n\n".join(lines)
