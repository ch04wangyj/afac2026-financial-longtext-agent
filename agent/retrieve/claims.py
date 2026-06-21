"""单选与多选共用的 Claim 级检索目标。"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.retrieve.structured_queries import extract_query_entities
from agent.schemas import Question


CLAIM_COMPARISON_HINTS = ("比较", "高于", "低于", "增加", "减少", "同比", "环比", "超过", "不超过")
CLAIM_CLAUSE_HINTS = ("处罚", "罚款", "扣减", "减分", "责令", "不得", "期限", "应当", "可以")
CLAIM_METRIC_HINTS = ("营业收入", "净利润", "现金流", "分红", "每股", "派现", "股息", "资产负债率")
CLAIM_DATE_HINTS = ("日期", "时间", "期限", "之前", "之后", "年度", "年末")


@dataclass(frozen=True)
class ClaimTarget:
    claim_id: str
    question_id: str
    option_key: str
    option_text: str
    source_question: str
    claim_text: str
    answer_format: str
    domain: str
    doc_scope: list[str] = field(default_factory=list)
    claim_type: str = "fact"
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    evidence_slots: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "question_id": self.question_id,
            "option_key": self.option_key,
            "option_text": self.option_text,
            "source_question": self.source_question,
            "claim_text": self.claim_text,
            "answer_format": self.answer_format,
            "domain": self.domain,
            "doc_scope": list(self.doc_scope),
            "claim_type": self.claim_type,
            "entities": list(self.entities),
            "numbers": list(self.numbers),
            "dates": list(self.dates),
            "must_terms": list(self.must_terms),
            "should_terms": list(self.should_terms),
            "evidence_slots": dict(self.evidence_slots),
        }


def build_claim_targets(question: Question) -> list[ClaimTarget]:
    claims: list[ClaimTarget] = []
    for option_key, option_text in sorted(question.options.items()):
        claim_text = f"{question.question}\n判断选项{option_key}是否正确：{option_text}".strip()
        combined = f"{question.question} {option_text}".strip()
        option_entities = extract_query_entities(option_text)
        question_entities = extract_query_entities(question.question)
        # Claim 检索首先服务于当前选项。题干模板词放在前面会截断真正的指标、
        # 主体和比较端点，因此实体与 must_terms 都按 option-first 组织。
        entities = _dedupe([*option_entities, *question_entities])[:12]
        numbers = _dedupe(extract_numbers(combined))[:6]
        dates = _dedupe(extract_dates(combined))[:6]
        claim_type = infer_claim_type(combined, numbers=numbers, dates=dates)
        must_terms = _dedupe([*option_entities[:6], *numbers[:4], *dates[:4], *question_entities[:3]])[:10]
        should_terms = _dedupe([option_text, *extract_query_entities(option_text)[:6]])[:8]
        claims.append(
            ClaimTarget(
                claim_id=f"{question.qid}:{option_key}",
                question_id=question.qid,
                option_key=option_key,
                option_text=option_text,
                source_question=question.question,
                claim_text=claim_text,
                answer_format=question.answer_format,
                domain=question.domain,
                doc_scope=list(question.doc_ids),
                claim_type=claim_type,
                entities=entities,
                numbers=numbers,
                dates=dates,
                must_terms=must_terms,
                should_terms=should_terms,
                evidence_slots=_initial_evidence_slots(claim_type),
            )
        )
    return claims


def infer_claim_type(text: str, *, numbers: list[str], dates: list[str]) -> str:
    if any(hint in text for hint in CLAIM_COMPARISON_HINTS):
        return "comparison"
    if any(hint in text for hint in CLAIM_CLAUSE_HINTS):
        return "clause_consequence"
    if any(hint in text for hint in CLAIM_METRIC_HINTS) or numbers:
        return "metric_fact"
    if any(hint in text for hint in CLAIM_DATE_HINTS) or dates:
        return "date_fact"
    return "fact"


def _initial_evidence_slots(claim_type: str) -> dict:
    common = {"entity_anchor": {"required": True}}
    if claim_type == "comparison":
        return {**common, "comparator_endpoint": {"required": True}, "value": {"required": True}}
    if claim_type == "clause_consequence":
        return {**common, "consequence": {"required": True}}
    if claim_type == "metric_fact":
        return {**common, "metric": {"required": True}, "value": {"required": True}}
    if claim_type == "date_fact":
        return {**common, "date": {"required": True}}
    return common



def claim_to_retrieval_target(claim: ClaimTarget):
    from agent.retrieve.targets import RetrievalTarget

    evidence_intent = "fact"
    if claim.claim_type == "comparison":
        evidence_intent = "comparison"
    elif claim.claim_type in {"metric_fact", "date_fact"}:
        evidence_intent = "number"
    return RetrievalTarget(
        node_id=claim.claim_id,
        rank=0,
        question=claim.claim_text,
        doc_scope=list(claim.doc_scope),
        must_terms=list(claim.must_terms),
        should_terms=list(claim.should_terms),
        numbers=list(claim.numbers),
        dates=list(claim.dates),
        entities=list(claim.entities),
        option_terms=[claim.option_text, *claim.should_terms],
        evidence_intent=evidence_intent,
        query_variants=[],
    )



def _dedupe(items) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(str(item or "").split())
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output
