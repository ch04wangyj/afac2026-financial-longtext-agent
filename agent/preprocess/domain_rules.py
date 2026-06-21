"""基于 Docling 样本生成各领域清洗/索引规则草案。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json


def infer_candidate_rules(domain: str, sample_text: str) -> dict:
    cleaning_rules: list[str] = []
    index_rules: list[str] = []
    lowered = sample_text or ""

    if domain == "financial_reports":
        if "目录" in lowered:
            cleaning_rules.append("drop_toc_blocks")
        if any(token in lowered for token in ["营业收入", "净利润", "现金流", "资产负债率"]):
            index_rules.append("boost_financial_metrics")
        if any(token in lowered for token in ["同比", "环比", "%", "亿元", "万元"]):
            index_rules.append("normalize_financial_units")
    elif domain == "financial_contracts":
        if any(token in lowered for token in ["目 录", "目录"]):
            cleaning_rules.append("drop_toc_blocks")
        if any(token in lowered for token in ["释义", "公司声明", "风险提示"]):
            cleaning_rules.append("downweight_template_sections")
        if any(token in lowered for token in ["交易对方", "认购", "发行", "股东"]):
            index_rules.append("boost_transaction_entities")
    elif domain == "insurance":
        if any(token in lowered for token in ["责任免除", "保险责任", "释义"]):
            index_rules.append("boost_clause_sections")
        if any(token in lowered for token in ["第一条", "第二条", "第十条"]):
            index_rules.append("index_clause_ids")
    elif domain == "research":
        if any(token in lowered for token in ["摘要", "投资要点", "结论"]):
            index_rules.append("boost_summary_sections")
        if any(token in lowered for token in ["图", "表", "预测", "同比"]):
            index_rules.append("bind_chart_caption_context")
    elif domain == "regulatory":
        if any(token in lowered for token in ["决定", "公告", "附件"]):
            index_rules.append("boost_notice_titles")
        if any(token in lowered for token in ["中国证监会", "公司法", "施行"]):
            index_rules.append("index_agency_and_effective_date")

    return {
        "domain": domain,
        "cleaning_rules": cleaning_rules,
        "index_rules": index_rules,
    }


def summarize_sample_bundle(sample_dir: Path) -> dict:
    full_txt = (sample_dir / "full.txt").read_text(encoding="utf-8") if (sample_dir / "full.txt").exists() else ""
    pages = json.loads((sample_dir / "pages.json").read_text(encoding="utf-8")) if (sample_dir / "pages.json").exists() else []
    tables = json.loads((sample_dir / "tables.json").read_text(encoding="utf-8")) if (sample_dir / "tables.json").exists() else []
    figures = json.loads((sample_dir / "figures.json").read_text(encoding="utf-8")) if (sample_dir / "figures.json").exists() else []
    page_count = len(pages)
    low_text_pages = sum(1 for row in pages if len((row.get("text") or "").strip()) < 80)
    nonempty_table_pages = sum(1 for row in pages if row.get("tables"))
    nonempty_figure_pages = sum(1 for row in pages if row.get("figures"))
    return {
        "page_count": page_count,
        "low_text_pages": low_text_pages,
        "table_items": len(tables),
        "figure_items": len(figures),
        "table_pages": nonempty_table_pages,
        "figure_pages": nonempty_figure_pages,
        "char_count": len(full_txt),
    }


def top_signals(sample_text: str, limit: int = 15) -> list[str]:
    tokens = [line.strip() for line in sample_text.splitlines() if line.strip()]
    counts = Counter(tokens)
    return [text for text, _ in counts.most_common(limit)]
