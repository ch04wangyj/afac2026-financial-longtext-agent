"""V13 选项验证查询：同时搜索支持陈述和不带候选值的真实谓词。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent.retrieve.claims import ClaimTarget
from agent.retrieve.structured_queries import extract_query_entities
from agent.schemas import Question


RELATION_TERMS = (
    "发行主体",
    "发行人",
    "发行规模",
    "募集资金",
    "受托管理人",
    "主承销商",
    "信用评级",
    "票面利率",
    "保险责任",
    "保险金",
    "等待期",
    "犹豫期",
    "现金价值",
    "退保",
    "免责",
    "赔付",
    "应当",
    "不得",
    "可以",
    "处罚",
    "罚款",
    "期限",
    "营业收入",
    "归母净利润",
    "归属于上市公司股东的净利润",
    "净利润",
    "经营活动现金流量净额",
    "经营活动现金流净额",
    "经营活动现金流",
    "经营现金流",
    "现金流量净额",
    "研发投入",
    "研发投入强度",
    "研发投入占比",
    "资产负债率",
    "每股现金分红",
    "现金分红总额",
    "分红",
    "股息",
    "同比",
    "发布日期",
    "施行日期",
    "成立时间",
    "股东会职权",
    "股东大会职权",
    "股东大会",
    "股东会",
    "董事会",
    "董事候选人",
    "担保",
    "书面形式",
    "表决",
    "批准",
    "审议批准",
)
REFERENCE_TERMS = {
    "第一份文档",
    "第二份文档",
    "该文档",
    "两份文档",
    "下列说法",
    "正确选项",
    "错误选项",
}
RATING_RE = re.compile(r"(?<![A-Za-z])(?:AAA|AA\+?|A\+?|BBB\+?)(?![A-Za-z])", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationQueryBundle:
    query: str
    intent: str
    weight: float


def build_verification_query_bundles(
    question: Question,
    claim: ClaimTarget,
    *,
    max_bundles: int = 6,
) -> list[VerificationQueryBundle]:
    """构造支持查询、真实值查询和条件例外查询。"""
    predicates = extract_predicate_terms(question, claim)
    candidate_values = extract_candidate_values(claim)
    option_entities = _clean_terms(extract_query_entities(claim.option_text))
    bundles = [
        VerificationQueryBundle(_join(*predicates, *candidate_values), "support", 1.0),
        # 不携带选项声称的数字/日期，避免错误选项把检索带向不存在的值。
        VerificationQueryBundle(_join(*predicates), "predicate_truth", 2.4),
    ]
    if option_entities:
        bundles.append(VerificationQueryBundle(_join(*predicates, *option_entities[:6]), "entity_support", 0.9))
    if claim.claim_type in {"clause_consequence", "date_fact"}:
        bundles.append(
            VerificationQueryBundle(
                _join(*predicates, "但", "除外", "仅限", "不得", "应当", "期限"),
                "exception_scope",
                1.2,
            )
        )
    if claim.claim_type in {"metric_fact", "comparison"}:
        bundles.append(
            VerificationQueryBundle(
                _join(*predicates, "单位", "本期", "上期", "同比", "合计"),
                "metric_ground_truth",
                1.4,
            )
        )
    return _dedupe_bundles(bundles)[:max_bundles]


def extract_predicate_terms(question: Question, claim: ClaimTarget) -> list[str]:
    """提取需要在原文中核验的关系，不把候选答案值当成谓词。"""
    # 题干常同时罗列多个主题；先按当前选项取谓词，避免“分红”污染净利润/现金流选项。
    terms = _matched_relation_terms(claim.option_text)
    if not terms:
        terms = _matched_relation_terms(question.question)
    if not terms:
        terms.extend(
            term
            for term in _clean_terms(extract_query_entities(claim.option_text))
            if term not in extract_candidate_values(claim) and not _looks_like_candidate_literal(term)
        )
    return _expand_relation_aliases(list(dict.fromkeys(terms)))[:16]


def extract_candidate_values(claim: ClaimTarget) -> list[str]:
    """候选值仅用于支持证据打分，不作为真实值查询的必要条件。"""
    values = [value for value in claim.numbers if not _is_plain_year(value)]
    values.extend(value for value in claim.dates if not _is_plain_year(value))
    values.extend(match.group(0).upper() for match in RATING_RE.finditer(claim.option_text))
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))[:8]


def _clean_terms(terms: list[str]) -> list[str]:
    output: list[str] = []
    for term in terms:
        term = " ".join(str(term).split())
        if not term or term in REFERENCE_TERMS or len(term) > 36:
            continue
        if any(reference in term for reference in REFERENCE_TERMS):
            continue
        output.append(term)
    return list(dict.fromkeys(output))


def _looks_like_candidate_literal(value: str) -> bool:
    return bool(re.fullmatch(r"[-+()（）\d\s,，.]+(?:%|％|元|万元|亿元|万|亿|年|月|日)?", value))


def _is_plain_year(value: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d{2}年?", re.sub(r"\s+", "", str(value or ""))))


def _matched_relation_terms(text: str) -> list[str]:
    """优先保留最长指标名，避免“归母净利润/净利润”重复计分。"""
    output: list[str] = []
    for term in sorted(RELATION_TERMS, key=len, reverse=True):
        if term not in text or any(term in existing for existing in output):
            continue
        output.append(term)
    return output


def _expand_relation_aliases(terms: list[str]) -> list[str]:
    """补充同一披露字段的常见写法，仍保持纯规则、无 embedding。"""
    joined = " ".join(terms)
    aliases: list[str] = []
    any_rules = (
        (("营业收入",), ("营业总收入", "营业额")),
        (("研发投入强度", "研发投入占比", "研发投入占营业收入比例"), ("研发投入占营业收入比例", "研发费用占营业收入比例")),
        (
            ("归母净利润", "归属于上市公司股东的净利润", "归属于母公司股东的净利润"),
            ("归属于上市公司股东的净利润", "归属于母公司的净利润", "母公司拥有人应占溢利"),
        ),
        (("经营活动现金流净额", "经营活动现金流", "经营活动现金流量净额"), ("经营活动产生的现金流量净额", "经营活动现金流量净额")),
        (("每股现金分红", "现金分红总额"), ("每10股派息", "每股股息", "末期股息", "利润分配方案")),
        (("股东大会",), ("股东会",)),
    )
    for triggers, values in any_rules:
        if any(trigger in joined for trigger in triggers):
            aliases.extend(values)
    if "董事会" in joined and "批准" in joined:
        aliases.extend(("董事会的报告", "董事会工作报告"))
    return list(dict.fromkeys([*terms, *aliases]))


def _join(*parts: str) -> str:
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _dedupe_bundles(bundles: list[VerificationQueryBundle]) -> list[VerificationQueryBundle]:
    output: list[VerificationQueryBundle] = []
    seen: set[str] = set()
    for bundle in bundles:
        query = " ".join(bundle.query.split())
        if query and query not in seen:
            output.append(VerificationQueryBundle(query, bundle.intent, bundle.weight))
            seen.add(query)
    return output
