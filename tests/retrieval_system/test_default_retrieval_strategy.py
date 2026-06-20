from __future__ import annotations

from agent.config import Settings
from agent.index.bm25 import BM25SearchIndex
from agent.retrieve.retriever import Retriever
from agent.schemas import Chunk, Question


def test_default_settings_strategy_uses_doc_first_bm25f_expansion_online_path():
    settings = Settings()
    assert settings.retrieval_strategy == "doc_first_bm25f_expansion"

    chunks = [
        Chunk(
            "c-target",
            "doc-midea",
            "financial_reports",
            1,
            "股东回报",
            "",
            "公司自2019年起连续实施股份回购计划，用于股权激励及员工持股计划。",
            [],
            ["2019 年"],
            [],
            {
                "title": "annual_midea_2024_report",
                "extra_index_fields": [
                    "美的集团",
                    "股份回购",
                    "2019 年",
                    "结论重述: 美的 2019 年 连续实施股份回购方案",
                ],
            },
        ),
        Chunk(
            "c-other",
            "doc-other",
            "financial_reports",
            1,
            "股权激励",
            "",
            "公司继续推进限制性股票激励计划并进行回购注销。",
            [],
            ["2025 年"],
            [],
            {"title": "annual_other_2025_report", "extra_index_fields": ["股权激励", "回购注销"]},
        ),
    ]
    index = BM25SearchIndex.build(chunks, tokenizer_mode="mixed")
    retriever = Retriever(index, strategy=settings.retrieval_strategy)
    question = Question(
        qid="q-default-strategy",
        domain="financial_reports",
        split="A",
        question="美的集团自2019年起连续实施了股份回购方案。",
        options={"A": "正确", "B": "错误"},
        answer_format="mcq",
        doc_ids=["doc-midea"],
    )

    results = retriever.retrieve(question)

    assert results
    assert results[0].doc_id == "doc-midea"
    assert results[0].source == "doc_first_bm25f_expansion"
