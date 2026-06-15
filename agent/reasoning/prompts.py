"""Qwen 作答 Prompt 模板。"""

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
