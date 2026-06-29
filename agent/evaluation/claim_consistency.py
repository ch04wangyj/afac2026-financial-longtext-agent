"""跨题重复事实一致性审计。

公开题集中存在复用同一文档事实、仅改变问法的题目。该模块把题目选项转成
轻量事实节点，并在共享文档、数字签名一致的前提下查找高相似断言的答案冲突。
它只生成复核候选，不自动修改答案，避免把否定词或口径差异误当成等价事实。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import combinations

from agent.schemas import AnswerResult, Question


_PUNCT_RE = re.compile(r"[\s，。；：、？！,!?;:（）()【】\[\]《》“”\"'·—\-_/]+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?\s*"
    r"(?:万亿元|亿元人民币|亿美元|亿欧元|亿元|万元|个百分点|个工作日|个自然日|"
    r"个月|倍|元|年|日|%)?"
)
_CHINESE_NUMBER_RE = re.compile(
    r"[一二三四五六七八九十百千万两]+"
    r"(?:个工作日|个自然日|个月|万元|年|日|倍)"
)
_QUALIFIER_CLASSES = (
    ("non_major", ("非重大差异",)),
    ("major", ("重大差异",)),
    ("manual", ("手动",)),
    ("automatic", ("自动",)),
    ("down", ("负增长", "下降", "下滑", "回落")),
    ("up", ("正增长", "增长", "提升", "上升")),
    ("before", ("早于",)),
    ("after", ("晚于",)),
    ("higher", ("高于", "高出", "超过")),
    ("lower", ("低于", "低出", "不足")),
    ("usd", ("美元",)),
    ("eur", ("欧元",)),
    ("cny", ("人民币",)),
    ("not_eligible", ("不具备", "不符合")),
    ("eligible", ("具备", "符合")),
    ("only", ("仅", "只保障", "只包括")),
    ("rating_aaa", ("AAA",)),
    ("rating_aa_plus", ("AA+",)),
)
_UNIVERSAL_SCOPE_MARKERS = ("两份", "两家", "双方", "均", "全部", "各自")
_TF_PREFIXES = (
    "判断以下陈述是否正确",
    "判断题",
    "根据文档信息",
    "根据相关监管规定",
)


@dataclass(frozen=True)
class ClaimRecord:
    """一个可比较的题目断言及当前答案状态。"""

    qid: str
    option: str
    domain: str
    text: str
    normalized_text: str
    selected: bool
    doc_ids: tuple[str, ...]
    source_doc_ids: tuple[str, ...]
    source_scoped: bool
    numbers: tuple[str, ...]
    qualifiers: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClaimConflict:
    """两个高相似断言在当前答案中具有相反真值。"""

    left: ClaimRecord
    right: ClaimRecord
    similarity: float
    shared_doc_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "similarity": round(self.similarity, 4),
            "shared_doc_ids": list(self.shared_doc_ids),
        }


def normalize_claim(text: str) -> str:
    """统一全半角、常见时间写法和标点，保留否定词与关键实体。"""
    value = unicodedata.normalize("NFKC", text).lower()
    value = value.replace("q1-3", "前三季度").replace("q1–3", "前三季度")
    value = value.replace("同比增速", "同比").replace("累计同比增速", "累计同比")
    value = _PUNCT_RE.sub("", value)
    return re.sub(r"(?<!\d)\.|\.(?!\d)", "", value)


def numeric_signature(text: str) -> tuple[str, ...]:
    """提取排序后的数字签名，降低相似句中不同年份或数值造成的误报。"""
    normalized = unicodedata.normalize("NFKC", text)
    values = [re.sub(r"\s+", "", item) for item in _NUMBER_RE.findall(normalized)]
    values.extend(_CHINESE_NUMBER_RE.findall(normalized))
    return tuple(sorted(values))


def qualifier_signature(text: str) -> tuple[str, ...]:
    """提取会改变事实真值的方向词、范围词和币种词。"""
    normalized = normalize_claim(text)
    values: list[str] = []
    for label, terms in _QUALIFIER_CLASSES:
        if any(term.lower() in normalized for term in terms):
            values.append(label)
    # “非重大差异”含有“重大差异”子串，只保留更具体的范围标签。
    if "non_major" in values and "major" in values:
        values.remove("major")
    if "rating_aa_plus" in values and "rating_aaa" in values:
        values.remove("rating_aaa")
    return tuple(values)


def build_claim_records(
    questions: list[Question],
    answers: list[AnswerResult],
    *,
    support_results: list[AnswerResult] | None = None,
) -> list[ClaimRecord]:
    """把多选/单选选项与判断题陈述转换为统一事实记录。"""
    answer_by_qid = {row.qid: row.answer for row in answers}
    support_by_qid = {row.qid: row for row in (support_results or [])}
    records: list[ClaimRecord] = []
    for question in questions:
        answer = answer_by_qid.get(question.qid)
        if answer is None:
            continue
        support_row = support_by_qid.get(question.qid)

        if question.answer_format == "tf":
            claim_text = _strip_tf_wrapper(question.question)
            records.append(
                _make_record(
                    question,
                    option="TF",
                    text=claim_text,
                    selected=answer == "A",
                    source_docs=_source_docs_for_option(support_row, "A"),
                )
            )
            continue

        for option, text in question.options.items():
            records.append(
                _make_record(
                    question,
                    option=option,
                    text=text,
                    selected=option in answer,
                    source_docs=_source_docs_for_option(support_row, option),
                )
            )
    return records


def find_claim_conflicts(
    records: list[ClaimRecord],
    *,
    min_similarity: float = 0.78,
    require_same_doc_set: bool = True,
) -> list[ClaimConflict]:
    """查找共享文档且数字口径一致的近重复断言答案冲突。"""
    conflicts: list[ClaimConflict] = []
    for left, right in combinations(records, 2):
        if left.qid == right.qid or left.domain != right.domain:
            continue
        # “正确”“以上说法”等短模板不包含足够事实信息，不能构成一致性证据。
        if min(len(left.normalized_text), len(right.normalized_text)) < 8:
            continue
        shared_doc_ids = tuple(sorted(set(left.source_doc_ids) & set(right.source_doc_ids)))
        if not shared_doc_ids or left.selected == right.selected:
            continue
        both_source_scoped = left.source_scoped and right.source_scoped
        if require_same_doc_set and not both_source_scoped and set(left.doc_ids) != set(right.doc_ids):
            continue
        if (
            _has_universal_scope(left.text)
            or _has_universal_scope(right.text)
        ) and set(left.source_doc_ids) != set(right.source_doc_ids):
            continue
        if left.numbers and right.numbers and left.numbers != right.numbers:
            continue
        if left.qualifiers != right.qualifiers:
            continue

        similarity = claim_similarity(left.normalized_text, right.normalized_text)
        if similarity < min_similarity:
            continue
        conflicts.append(
            ClaimConflict(
                left=left,
                right=right,
                similarity=similarity,
                shared_doc_ids=shared_doc_ids,
            )
        )
    return sorted(conflicts, key=lambda item: (-item.similarity, item.left.qid, item.right.qid))


def claim_similarity(left: str, right: str) -> float:
    """结合编辑相似度与二元字符集合，兼顾短句和局部改写。"""
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right))

    sequence_score = SequenceMatcher(None, left, right).ratio()
    left_grams = _char_ngrams(left)
    right_grams = _char_ngrams(right)
    union = left_grams | right_grams
    jaccard_score = len(left_grams & right_grams) / len(union) if union else 0.0
    return max(sequence_score, jaccard_score)


def _make_record(
    question: Question,
    *,
    option: str,
    text: str,
    selected: bool,
    source_docs: tuple[str, ...],
) -> ClaimRecord:
    source_scoped = bool(source_docs)
    return ClaimRecord(
        qid=question.qid,
        option=option,
        domain=question.domain,
        text=text,
        normalized_text=normalize_claim(text),
        selected=selected,
        doc_ids=tuple(question.doc_ids),
        source_doc_ids=source_docs or tuple(question.doc_ids),
        source_scoped=source_scoped,
        numbers=numeric_signature(text),
        qualifiers=qualifier_signature(text),
    )


def _strip_tf_wrapper(text: str) -> str:
    value = text.strip()
    for prefix in _TF_PREFIXES:
        value = value.removeprefix(prefix).lstrip("：:，, ")
    return value


def _char_ngrams(text: str, size: int = 2) -> set[str]:
    """生成字符 n-gram 集合；极短文本退化为整句集合。"""
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _source_docs_for_option(
    result: AnswerResult | None,
    option: str,
) -> tuple[str, ...]:
    """从 V6 证据契约读取选项谓词实际命中的文档范围。"""
    if result is None:
        return ()
    report = result.metadata.get("retrieval_report", {})
    contracts = report.get("evidence_contracts", {}) if isinstance(report, dict) else {}
    contract = contracts.get(option, {}) if isinstance(contracts, dict) else {}
    doc_ids = contract.get("predicate_doc_ids", []) if isinstance(contract, dict) else []
    return tuple(dict.fromkeys(str(doc_id) for doc_id in doc_ids if str(doc_id)))


def _has_universal_scope(text: str) -> bool:
    if any(marker in text for marker in _UNIVERSAL_SCOPE_MARKERS):
        return True
    return "所有" in text.replace("受益所有人", "")
