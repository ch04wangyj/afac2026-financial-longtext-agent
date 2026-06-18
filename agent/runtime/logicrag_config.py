"""LogicRAG / Qwen runtime YAML 配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "logicrag_runtime.yaml"


@dataclass(frozen=True)
class ThinkingProfile:
    """单个推理步骤的 thinking 开关与 token 预算。"""

    enabled: bool = True
    max_tokens: int = 512


@dataclass(frozen=True)
class QwenRuntimeConfig:
    """Qwen API 相关默认配置。"""

    model: str = "qwen3.7-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    request_timeout_seconds: int = 120
    max_retries: int = 2


@dataclass(frozen=True)
class LogicRAGSection:
    """LogicRAG 主流程参数。"""

    enabled: bool = True
    max_subproblems: int = 6
    max_ranks: int = 4
    rank_top_k: int = 12
    memory_chars: int = 4500


@dataclass(frozen=True)
class ConcurrencyConfig:
    """并发执行参数。"""

    question_workers: int = 15
    qwen_workers: int = 8
    qwen_request_limit: int = 100
    bm25_workers: int = 8


@dataclass(frozen=True)
class LogicRAGRuntimeConfig:
    """运行时总配置。"""

    qwen: QwenRuntimeConfig = field(default_factory=QwenRuntimeConfig)
    thinking_profiles: dict[str, ThinkingProfile] = field(default_factory=dict)
    logicrag: LogicRAGSection = field(default_factory=LogicRAGSection)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    source_path: Path = DEFAULT_CONFIG_PATH


def load_logicrag_runtime_config(path: Path | None = None) -> LogicRAGRuntimeConfig:
    """加载 YAML 配置，并允许环境变量覆写关键字段。"""
    config_path = Path(os.getenv("AFAC_LOGICRAG_CONFIG") or path or DEFAULT_CONFIG_PATH)
    raw = _read_yaml(config_path)

    qwen = QwenRuntimeConfig(
        model=str((raw.get("qwen") or {}).get("model", QwenRuntimeConfig.model)),
        base_url=str((raw.get("qwen") or {}).get("base_url", QwenRuntimeConfig.base_url)),
        request_timeout_seconds=int(
            (raw.get("qwen") or {}).get("request_timeout_seconds", QwenRuntimeConfig.request_timeout_seconds)
        ),
        max_retries=int((raw.get("qwen") or {}).get("max_retries", QwenRuntimeConfig.max_retries)),
    )

    logicrag = LogicRAGSection(
        enabled=_as_bool((raw.get("logicrag") or {}).get("enabled", LogicRAGSection.enabled)),
        max_subproblems=int((raw.get("logicrag") or {}).get("max_subproblems", LogicRAGSection.max_subproblems)),
        max_ranks=int((raw.get("logicrag") or {}).get("max_ranks", LogicRAGSection.max_ranks)),
        rank_top_k=int((raw.get("logicrag") or {}).get("rank_top_k", LogicRAGSection.rank_top_k)),
        memory_chars=int((raw.get("logicrag") or {}).get("memory_chars", LogicRAGSection.memory_chars)),
    )

    concurrency = ConcurrencyConfig(
        question_workers=int(
            (raw.get("concurrency") or {}).get("question_workers", ConcurrencyConfig.question_workers)
        ),
        qwen_workers=int((raw.get("concurrency") or {}).get("qwen_workers", ConcurrencyConfig.qwen_workers)),
        qwen_request_limit=int(
            (raw.get("concurrency") or {}).get("qwen_request_limit", ConcurrencyConfig.qwen_request_limit)
        ),
        bm25_workers=int((raw.get("concurrency") or {}).get("bm25_workers", ConcurrencyConfig.bm25_workers)),
    )

    thinking_profiles = {
        name: ThinkingProfile(
            enabled=_as_bool((data or {}).get("enabled", ThinkingProfile.enabled)),
            max_tokens=int((data or {}).get("max_tokens", ThinkingProfile.max_tokens)),
        )
        for name, data in (raw.get("thinking_profiles") or {}).items()
    }
    thinking_profiles = _with_default_profiles(thinking_profiles)

    config = LogicRAGRuntimeConfig(
        qwen=qwen,
        thinking_profiles=thinking_profiles,
        logicrag=logicrag,
        concurrency=concurrency,
        source_path=config_path,
    )
    return _apply_env_overrides(config)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}


def _with_default_profiles(profiles: dict[str, ThinkingProfile]) -> dict[str, ThinkingProfile]:
    defaults = {
        "answer_single_pass": ThinkingProfile(enabled=False, max_tokens=384),
        "logicrag_planner": ThinkingProfile(enabled=True, max_tokens=1024),
        "logicrag_rank_summary": ThinkingProfile(enabled=True, max_tokens=640),
        "logicrag_final_compose": ThinkingProfile(enabled=True, max_tokens=1024),
        "option_judgement": ThinkingProfile(enabled=False, max_tokens=192),
        "multi_option_fallback": ThinkingProfile(enabled=True, max_tokens=512),
    }
    merged = dict(defaults)
    merged.update(profiles)
    return merged


def _apply_env_overrides(config: LogicRAGRuntimeConfig) -> LogicRAGRuntimeConfig:
    qwen = QwenRuntimeConfig(
        model=os.getenv("AFAC_LOGICRAG_QWEN_MODEL", config.qwen.model),
        base_url=os.getenv("AFAC_LOGICRAG_QWEN_BASE_URL", config.qwen.base_url),
        request_timeout_seconds=int(
            os.getenv("AFAC_LOGICRAG_REQUEST_TIMEOUT_SECONDS", str(config.qwen.request_timeout_seconds))
        ),
        max_retries=int(os.getenv("AFAC_LOGICRAG_MAX_RETRIES", str(config.qwen.max_retries))),
    )
    logicrag = LogicRAGSection(
        enabled=_as_bool(os.getenv("AFAC_LOGICRAG_ENABLED", str(config.logicrag.enabled))),
        max_subproblems=int(os.getenv("AFAC_LOGICRAG_MAX_SUBPROBLEMS", str(config.logicrag.max_subproblems))),
        max_ranks=int(os.getenv("AFAC_LOGICRAG_MAX_RANKS", str(config.logicrag.max_ranks))),
        rank_top_k=int(os.getenv("AFAC_LOGICRAG_RANK_TOP_K", str(config.logicrag.rank_top_k))),
        memory_chars=int(os.getenv("AFAC_LOGICRAG_MEMORY_CHARS", str(config.logicrag.memory_chars))),
    )
    concurrency = ConcurrencyConfig(
        question_workers=int(os.getenv("AFAC_LOGICRAG_QUESTION_WORKERS", str(config.concurrency.question_workers))),
        qwen_workers=int(os.getenv("AFAC_LOGICRAG_QWEN_WORKERS", str(config.concurrency.qwen_workers))),
        qwen_request_limit=int(
            os.getenv("AFAC_LOGICRAG_QWEN_REQUEST_LIMIT", str(config.concurrency.qwen_request_limit))
        ),
        bm25_workers=int(os.getenv("AFAC_LOGICRAG_BM25_WORKERS", str(config.concurrency.bm25_workers))),
    )

    profiles = dict(config.thinking_profiles)
    for name, profile in list(profiles.items()):
        prefix = f"AFAC_LOGICRAG_PROFILE_{_normalize_name(name)}"
        enabled = _as_bool(os.getenv(f"{prefix}_ENABLED", str(profile.enabled)))
        max_tokens = int(os.getenv(f"{prefix}_MAX_TOKENS", str(profile.max_tokens)))
        profiles[name] = ThinkingProfile(enabled=enabled, max_tokens=max_tokens)

    return LogicRAGRuntimeConfig(
        qwen=qwen,
        thinking_profiles=profiles,
        logicrag=logicrag,
        concurrency=concurrency,
        source_path=config.source_path,
    )


def _normalize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).upper()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
