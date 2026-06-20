from agent.reasoning.prompts import (
    build_answer_messages,
    build_logicrag_final_compose_messages,
    build_logicrag_plan_messages,
)
from agent.schemas import LogicNode, LogicPlan, Question, RetrievalResult


def _question() -> Question:
    return Question(
        qid="q_prompt_contract",
        domain="financial_reports",
        split="A",
        question="根据财报披露，以下哪些说法正确？",
        options={
            "A": "2025 年收入增长",
            "B": "2025 年收入下降",
        },
        answer_format="multi",
        doc_ids=["annual_report_2025"],
    )


def _evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="annual:1",
            doc_id="annual_report_2025",
            domain="financial_reports",
            score=1.0,
            source="test",
            query="收入增长",
            evidence_text="2025 年公司营业收入同比增长 12%。",
            metadata={"title": "2025 年年报", "page": 3},
        )
    ]


def _logic_plan() -> LogicPlan:
    return LogicPlan(
        nodes=[
            LogicNode(node_id="n1", text="定位营业收入同比数据", depends_on=[]),
            LogicNode(node_id="n2", text="判断选项 A 与 B 哪个被证据支持", depends_on=["n1"]),
        ],
        rationale="先定位指标，再判断选项。",
    )


def test_answer_prompt_forbids_broad_token_heavy_compensation_for_missing_evidence():
    text = build_answer_messages(_question(), _evidence())[-1]["content"]

    assert "不要为了弥补检索不足而展开泛化推理" in text
    assert "不要输出冗长背景解释来补偿证据缺口" in text
    assert "证据不足时仅可降低 confidence" in text


def test_logicrag_plan_prompt_requires_precise_retrieval_targets_instead_of_background_expansion():
    text = build_logicrag_plan_messages(_question(), max_subproblems=4, max_ranks=3)[-1]["content"]

    assert "每个子问题必须指向可直接检索的具体事实、条款、数值或定义" in text
    assert "不要把背景介绍、概念解释或大范围综述写成子问题" in text


def test_logicrag_final_compose_prompt_stays_summary_first_and_rejects_token_heavy_compensation():
    text = build_logicrag_final_compose_messages(
        _question(),
        _evidence(),
        _logic_plan(),
        [{"rank": 0, "summary": "已确认 2025 年营业收入同比增长 12%。"}],
    )[-1]["content"]

    assert "不要为了补偿上游检索缺口而扩写背景或常识推理" in text
    assert "若分层记忆与最终证据仍不足，只能降低 confidence" in text
    assert "分层记忆优先于最终层原文堆砌" in text
