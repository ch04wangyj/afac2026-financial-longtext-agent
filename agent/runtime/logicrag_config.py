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

    level: str = "medium"
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
    execution_mode: str = "paper_faithful_core"
    retrieval_backend: str = "paper_contract_embedding"
    dynamic_dag_augmentation: bool = True
    append_unresolved_after_current_rank: bool = True
    sampling_without_replacement: bool = True
    max_subproblems: int = 6
    max_ranks: int = 4
    rank_top_k: int = 5
    memory_chars: int = 4500


@dataclass(frozen=True)
class ABoardRuntimeConfig:
    """A 榜质量模式参数。"""

    option_matrix_enabled: bool = False
    multi_logicrag_enabled: bool = True
    multi_logicrag_retry_enabled: bool = True
    coverage_gate_enabled: bool = False
    force_doc_coverage_for_a_board: bool = True
    use_doc_ids_as_hint_only: bool = False
    financial_calculator_enabled: bool = False
    max_option_candidates: int = 12
    max_verifier_candidates_per_option: int = 6
    low_confidence_threshold: float = 0.65


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
    a_board: ABoardRuntimeConfig = field(default_factory=ABoardRuntimeConfig)
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
        execution_mode=str((raw.get("logicrag") or {}).get("execution_mode", LogicRAGSection.execution_mode)),
        retrieval_backend=str((raw.get("logicrag") or {}).get("retrieval_backend", LogicRAGSection.retrieval_backend)),
        dynamic_dag_augmentation=_as_bool(
            (raw.get("logicrag") or {}).get("dynamic_dag_augmentation", LogicRAGSection.dynamic_dag_augmentation)
        ),
        append_unresolved_after_current_rank=_as_bool(
            (raw.get("logicrag") or {}).get(
                "append_unresolved_after_current_rank", LogicRAGSection.append_unresolved_after_current_rank
            )
        ),
        sampling_without_replacement=_as_bool(
            (raw.get("logicrag") or {}).get(
                "sampling_without_replacement", LogicRAGSection.sampling_without_replacement
            )
        ),
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

    a_board = ABoardRuntimeConfig(
        option_matrix_enabled=_as_bool(
            (raw.get("a_board") or {}).get("option_matrix_enabled", ABoardRuntimeConfig.option_matrix_enabled)
        ),
        multi_logicrag_enabled=_as_bool(
            (raw.get("a_board") or {}).get("multi_logicrag_enabled", ABoardRuntimeConfig.multi_logicrag_enabled)
        ),
        multi_logicrag_retry_enabled=_as_bool(
            (raw.get("a_board") or {}).get("multi_logicrag_retry_enabled", ABoardRuntimeConfig.multi_logicrag_retry_enabled)
        ),
        coverage_gate_enabled=_as_bool(
            (raw.get("a_board") or {}).get("coverage_gate_enabled", ABoardRuntimeConfig.coverage_gate_enabled)
        ),
        force_doc_coverage_for_a_board=_as_bool(
            (raw.get("a_board") or {}).get(
                "force_doc_coverage_for_a_board", ABoardRuntimeConfig.force_doc_coverage_for_a_board
            )
        ),
        use_doc_ids_as_hint_only=_as_bool(
            (raw.get("a_board") or {}).get(
                "use_doc_ids_as_hint_only", ABoardRuntimeConfig.use_doc_ids_as_hint_only
            )
        ),
        financial_calculator_enabled=_as_bool(
            (raw.get("a_board") or {}).get(
                "financial_calculator_enabled", ABoardRuntimeConfig.financial_calculator_enabled
            )
        ),
        max_option_candidates=int(
            (raw.get("a_board") or {}).get("max_option_candidates", ABoardRuntimeConfig.max_option_candidates)
        ),
        max_verifier_candidates_per_option=int(
            (raw.get("a_board") or {}).get(
                "max_verifier_candidates_per_option", ABoardRuntimeConfig.max_verifier_candidates_per_option
            )
        ),
        low_confidence_threshold=float(
            (raw.get("a_board") or {}).get(
                "low_confidence_threshold", ABoardRuntimeConfig.low_confidence_threshold
            )
        ),
    )

    thinking_profiles = {
        name: ThinkingProfile(
            level=str((data or {}).get("level", ThinkingProfile.level)).strip().lower(),
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
        a_board=a_board,
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
        "answer_single_pass": ThinkingProfile(level="low", enabled=False, max_tokens=384),
        "logicrag_planner": ThinkingProfile(level="high", enabled=True, max_tokens=1024),
        "logicrag_rank_summary": ThinkingProfile(level="medium", enabled=True, max_tokens=640),
        "logicrag_final_compose": ThinkingProfile(level="high", enabled=True, max_tokens=1024),
        "option_judgement": ThinkingProfile(level="low", enabled=False, max_tokens=192),
        "multi_option_fallback": ThinkingProfile(level="medium", enabled=True, max_tokens=512),
        "multi_logicrag_option_planner": ThinkingProfile(level="low", enabled=False, max_tokens=160),
        "multi_logicrag_option_verdict": ThinkingProfile(level="low", enabled=False, max_tokens=192),
        "multi_logicrag_option_retry": ThinkingProfile(level="low", enabled=True, max_tokens=192),
        "multi_logicrag_numeric_verifier": ThinkingProfile(level="low", enabled=True, max_tokens=320),
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
        execution_mode=os.getenv("AFAC_LOGICRAG_EXECUTION_MODE", config.logicrag.execution_mode),
        retrieval_backend=os.getenv("AFAC_LOGICRAG_RETRIEVAL_BACKEND", config.logicrag.retrieval_backend),
        dynamic_dag_augmentation=_as_bool(
            os.getenv("AFAC_LOGICRAG_DYNAMIC_DAG_AUGMENTATION", str(config.logicrag.dynamic_dag_augmentation))
        ),
        append_unresolved_after_current_rank=_as_bool(
            os.getenv(
                "AFAC_LOGICRAG_APPEND_UNRESOLVED_AFTER_CURRENT_RANK",
                str(config.logicrag.append_unresolved_after_current_rank),
            )
        ),
        sampling_without_replacement=_as_bool(
            os.getenv(
                "AFAC_LOGICRAG_SAMPLING_WITHOUT_REPLACEMENT",
                str(config.logicrag.sampling_without_replacement),
            )
        ),
        max_subproblems=int(os.getenv("AFAC_LOGICRAG_MAX_SUBPROBLEMS", str(config.logicrag.max_subproblems))),
        max_ranks=int(os.getenv("AFAC_LOGICRAG_MAX_RANKS", str(config.logicrag.max_ranks))),
        rank_top_k=int(os.getenv("AFAC_LOGICRAG_RANK_TOP_K", str(config.logicrag.rank_top_k))),
        memory_chars=int(os.getenv("AFAC_LOGICRAG_MEMORY_CHARS", str(config.logicrag.memory_chars))),
    )
    a_board = ABoardRuntimeConfig(
        option_matrix_enabled=_as_bool(
            os.getenv("AFAC_A_BOARD_OPTION_MATRIX_ENABLED", str(config.a_board.option_matrix_enabled))
        ),
        multi_logicrag_enabled=_as_bool(
            os.getenv("AFAC_A_BOARD_MULTI_LOGICRAG_ENABLED", str(config.a_board.multi_logicrag_enabled))
        ),
        multi_logicrag_retry_enabled=_as_bool(
            os.getenv("AFAC_A_BOARD_MULTI_LOGICRAG_RETRY_ENABLED", str(config.a_board.multi_logicrag_retry_enabled))
        ),
        coverage_gate_enabled=_as_bool(
            os.getenv("AFAC_A_BOARD_COVERAGE_GATE_ENABLED", str(config.a_board.coverage_gate_enabled))
        ),
        force_doc_coverage_for_a_board=_as_bool(
            os.getenv(
                "AFAC_A_BOARD_FORCE_DOC_COVERAGE_FOR_A_BOARD",
                str(config.a_board.force_doc_coverage_for_a_board),
            )
        ),
        use_doc_ids_as_hint_only=_as_bool(
            os.getenv("AFAC_A_BOARD_USE_DOC_IDS_AS_HINT_ONLY", str(config.a_board.use_doc_ids_as_hint_only))
        ),
        financial_calculator_enabled=_as_bool(
            os.getenv(
                "AFAC_A_BOARD_FINANCIAL_CALCULATOR_ENABLED",
                str(config.a_board.financial_calculator_enabled),
            )
        ),
        max_option_candidates=int(
            os.getenv("AFAC_A_BOARD_MAX_OPTION_CANDIDATES", str(config.a_board.max_option_candidates))
        ),
        max_verifier_candidates_per_option=int(
            os.getenv(
                "AFAC_A_BOARD_MAX_VERIFIER_CANDIDATES_PER_OPTION",
                str(config.a_board.max_verifier_candidates_per_option),
            )
        ),
        low_confidence_threshold=float(
            os.getenv("AFAC_A_BOARD_LOW_CONFIDENCE_THRESHOLD", str(config.a_board.low_confidence_threshold))
        ),
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
        level = os.getenv(f"{prefix}_LEVEL", profile.level).strip().lower()
        enabled = _as_bool(os.getenv(f"{prefix}_ENABLED", str(profile.enabled)))
        max_tokens = int(os.getenv(f"{prefix}_MAX_TOKENS", str(profile.max_tokens)))
        profiles[name] = ThinkingProfile(level=level, enabled=enabled, max_tokens=max_tokens)

    return LogicRAGRuntimeConfig(
        qwen=qwen,
        thinking_profiles=profiles,
        logicrag=logicrag,
        a_board=a_board,
        concurrency=concurrency,
        source_path=config.source_path,
    )


def _normalize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).upper()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
