"""结构化查询构造器。

这些函数把 GraphRAG/LogicRAG/LinearRAG 的思想改写成无 embedding 的 lite 版本：
只做规则实体抽取和多查询 RRF，不引入额外模型或向量表示。
"""

from __future__ import annotations

import itertools
import re

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.schemas import Question


LAW_RE = re.compile(r"《[^》]{2,80}》")
YEAR_RE = re.compile(r"20\d{2}\s*年")
ENTITY_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9]{2,30}(?:公司|集团|银行|保险|证券|基金|债券|报告|办法|条例|规定|细则|通知|指引|准则|产品)"
)
METRIC_RE = re.compile(
    r"(?:营业收入|净利润|现金流量净额|研发投入|分红|票面利率|发行规模|评级|担保|身故保险金|现金价值|保险责任|退保|客户尽职调查|受益所有人)"
)


def extract_query_entities(text: str) -> list[str]:
    """从题干/选项中抽取法规名、年份、金融实体、指标、数字和日期。"""
    entities: list[str] = []
    for pattern in (LAW_RE, ENTITY_RE, METRIC_RE, YEAR_RE):
        entities.extend(match.group(0).strip() for match in pattern.finditer(text))
    entities.extend(extract_numbers(text))
    entities.extend(extract_dates(text))
    return _dedupe([item for item in entities if len(item.strip()) >= 2])[:24]


def build_logic_queries(question: Question) -> list[str]:
    """LogicRAG-lite：按题干和选项构造查询时子问题，不依赖预构图。"""
    stem_entities = extract_query_entities(question.question)
    queries = [question.question]
    if stem_entities:
        queries.append(f"{question.question} {' '.join(stem_entities)}")
    for key, option in sorted(question.options.items()):
        option_entities = extract_query_entities(option)
        queries.append(f"{question.question} {key} {option}")
        if option_entities:
            queries.append(f"{' '.join(stem_entities[:8])} {' '.join(option_entities[:8])}")
    return _dedupe(queries)[:12]


def build_linear_entity_queries(question: Question) -> list[str]:
    """LinearRAG-lite：把高信号实体线性展开，用于盲搜和实体题补召回。"""
    full_text = _full_question_text(question)
    entities = extract_query_entities(full_text)
    if not entities:
        return [_full_question_text(question)]
    queries = [" ".join(entities[:8])]
    queries.extend(f"{entity} {question.question}" for entity in entities[:12])
    return _dedupe(queries)[:12]


def build_graph_lite_queries(question: Question) -> list[str]:
    """GraphRAG-lite：用实体 pair 查询近似共现边，不生成社区摘要。"""
    full_text = _full_question_text(question)
    entities = extract_query_entities(full_text)
    queries = [_full_question_text(question)]
    queries.extend(f"{a} {b}" for a, b in itertools.combinations(entities[:8], 2))
    for key, option in sorted(question.options.items()):
        option_entities = extract_query_entities(option)
        for entity in option_entities[:4]:
            queries.append(f"{entity} {option}")
    return _dedupe(queries)[:16]


def _full_question_text(question: Question) -> str:
    """拼接题干和所有选项，作为完整查询文本。"""
    return f"{question.question} " + " ".join(
        f"{key} {value}" for key, value in sorted(question.options.items())
    )


def _dedupe(items: list[str]) -> list[str]:
    """按原始顺序去重，并清理多余空白。"""
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join(item.split())
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
