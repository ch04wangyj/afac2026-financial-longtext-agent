"""题干选择范围契约：区分“事实成立”和“应按题干入选”。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from agent.schemas import Question


_INCORRECT_PATTERNS = (
    r"不正确(?:的|的是)",
    r"错误(?:的|的是)",
    r"不符合(?:的是|下列)",
    r"不属于(?:的是|下列)",
    r"不包括(?:的是|下列)",
)
_SCOPE_MARKERS = (
    "审批",
    "报告",
    "报送",
    "备案",
    "金额",
    "门槛",
    "期限",
    "条件",
    "范围",
    "情形",
    "责任",
    "赔付",
    "除外",
    "指标",
    "增速",
    "高于",
    "低于",
)
_UNIVERSAL_MARKERS = ("均", "全部", "所有", "分别", "两份", "各")
_COMPARISON_MARKERS = ("高于", "低于", "超过", "不超过", "多于", "少于", "差额", "相比")
_SCOPE_QUERY_PATTERNS = (
    r"(?:哪些|哪项|哪一项).{0,36}(?:需要|应当|可以|能够|属于|满足|适用|赔付|审批)",
    r"(?:需要|应当|可以|能够|属于|满足|适用|赔付|审批).{0,36}(?:哪些|哪项|情形|产品)",
)


@dataclass(frozen=True)
class QuestionEnvelope:
    """模型裁决前固定的题干语义边界。"""

    selection_rule: str
    focus: str
    scope_markers: tuple[str, ...]
    scope_gate_enabled: bool
    requires_all_documents: bool
    requires_comparison: bool
    requires_compound_truth: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt(self) -> str:
        markers = "、".join(self.scope_markers) or "无显式类别词，按完整题干判断"
        return "\n".join(
            (
                f"selection_rule={self.selection_rule}",
                f"focus={self.focus}",
                f"scope_markers={markers}",
                f"scope_gate_enabled={str(self.scope_gate_enabled).lower()}",
                f"requires_all_documents={str(self.requires_all_documents).lower()}",
                f"requires_comparison={str(self.requires_comparison).lower()}",
                f"requires_compound_truth={str(self.requires_compound_truth).lower()}",
                "fact_truth=选项完整事实是否被原文支持；"
                + (
                    "applicable=该事实是否满足题干要求的类别、关系、主体和口径，而非仅仅主题相关。"
                    if self.scope_gate_enabled
                    else "本题没有独立集合范围门禁，禁止用 applicable 排除事实成立的选项。"
                ),
            )
        )


def build_question_envelope(question: Question) -> QuestionEnvelope:
    """从题面提取稳定的选择规则和范围提示，不让模型自行改写题意。"""
    text = _compact(question.question)
    selection_rule = (
        "incorrect"
        if any(re.search(pattern, text) for pattern in _INCORRECT_PATTERNS)
        else "correct"
    )
    scope_markers = tuple(marker for marker in _SCOPE_MARKERS if marker in text)
    scope_gate_enabled = any(
        re.search(pattern, text) for pattern in _SCOPE_QUERY_PATTERNS
    )
    requires_all_documents = len(question.doc_ids) > 1 and any(
        marker in text for marker in _UNIVERSAL_MARKERS
    )
    requires_comparison = any(marker in text for marker in _COMPARISON_MARKERS)
    requires_compound_truth = any(marker in text for marker in ("且", "同时", "并且", "以及"))
    return QuestionEnvelope(
        selection_rule=selection_rule,
        focus=_extract_focus(text),
        scope_markers=scope_markers,
        scope_gate_enabled=scope_gate_enabled,
        requires_all_documents=requires_all_documents,
        requires_comparison=requires_comparison,
        requires_compound_truth=requires_compound_truth,
    )


def selection_verdict(
    fact_truth: str,
    applicable: str,
    selection_rule: str,
) -> str:
    """用双层裁决确定是否入选；任一层不确定时不冒进。"""
    fact = str(fact_truth).strip().lower()
    scope = str(applicable).strip().lower()
    if fact not in {"true", "false", "uncertain"}:
        return ""
    if scope not in {"true", "false", "uncertain"}:
        return ""
    if scope == "false":
        return "false"
    if scope == "uncertain" or fact == "uncertain":
        return "uncertain"
    if selection_rule == "incorrect":
        return "true" if fact == "false" else "false"
    return "true" if fact == "true" else "false"


def _extract_focus(text: str) -> str:
    """删除通用问句外壳，保留题目真正要求判断的关系。"""
    focus = re.sub(r"^(?:根据|依据|结合).{0,80}?(?:，|,)", "", text)
    focus = re.sub(
        r"(?:下列|以下)(?:说法|表述|选项)?(?:中)?(?:哪些|哪项|哪一项)?",
        "",
        focus,
    )
    focus = re.sub(r"(?:正确|错误|不正确|符合|不符合)(?:的|的是)?[？?]?$", "", focus)
    return focus.strip("，,：:；;？? ") or text


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))
