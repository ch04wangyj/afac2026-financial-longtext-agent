from __future__ import annotations

LIVE_RUNTIME_STRATEGIES = {
    "doc_first_bm25f_expansion",
    "logicrag_agent",
    "logicrag_qwen_rrf",
}


def validate_runtime_strategy(name: str) -> str:
    if name not in LIVE_RUNTIME_STRATEGIES:
        raise ValueError(f"Unsupported runtime strategy: {name}")
    return name
