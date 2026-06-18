"""Option-level retrieval query expansion and fusion helpers."""

from __future__ import annotations

from agent.preprocess.chunkers import extract_dates, extract_numbers
from agent.retrieve.fusion import reciprocal_rank_fusion
from agent.retrieve.structured_queries import extract_query_entities
from agent.schemas import Question, RetrievalResult


def build_option_queries(question: Question, max_queries_per_option: int = 5) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    stem_entities = extract_query_entities(question.question)
    for key, option in sorted(question.options.items()):
        full = f"{question.question} {key} {option}".strip()
        option_entities = extract_query_entities(option)
        numbers = extract_numbers(full)
        dates = extract_dates(full)
        queries = [full]
        if option_entities:
            queries.append(f"{question.question} {' '.join(option_entities)}")
        if stem_entities or option_entities:
            queries.append(" ".join([*stem_entities[:8], *option_entities[:8]]))
        if numbers:
            queries.append(f"{question.question} {' '.join(numbers)}")
        if dates:
            queries.append(f"{question.question} {' '.join(dates)}")
        output[key] = _dedupe(queries)[:max_queries_per_option]
    return output


def retrieve_option_candidates(
    index,
    question: Question,
    filter_doc_ids: set[str] | None,
    top_k_per_query: int = 12,
    fused_top_k: int = 12,
) -> dict[str, list[RetrievalResult]]:
    output: dict[str, list[RetrievalResult]] = {}
    for option_key, queries in build_option_queries(question).items():
        ranked_lists = [
            index.search(
                query=query,
                top_k=top_k_per_query,
                filter_doc_ids=filter_doc_ids,
                source=f"option_{option_key}",
            )
            for query in queries
        ]
        output[option_key] = reciprocal_rank_fusion(ranked_lists, top_k=fused_top_k)
    return output


def _dedupe(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = " ".join(item.split())
        if item and item not in seen:
            output.append(item)
            seen.add(item)
    return output
