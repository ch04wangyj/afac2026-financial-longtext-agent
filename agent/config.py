"""项目配置入口：集中读取路径、Qwen 模型参数和本地密钥配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    # 允许本地通过 .env 配置密钥和路径；该文件已加入 .gitignore，避免误提交。
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


def _local_value(name: str) -> str | None:
    """从被 git 忽略的 agent/local_settings.py 读取本地私有配置。"""
    try:
        from agent import local_settings
    except Exception:
        return None
    value = getattr(local_settings, name, None)
    return str(value) if value not in (None, "") else None


def _setting(name: str, default: str) -> str:
    """读取普通字符串配置，优先级：环境变量 > local_settings > 默认值。"""
    return os.getenv(name) or _local_value(name) or default


def _env_path(name: str, default: Path) -> Path:
    """读取路径配置，并统一展开为绝对路径，避免脚本工作目录变化导致路径漂移。"""
    value = os.getenv(name) or _local_value(name)
    return Path(value).expanduser().resolve() if value else default


def _env_bool(name: str, default: bool) -> bool:
    """读取布尔配置，支持常见的 true/yes/on 写法。"""
    value = os.getenv(name) or _local_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    """全局运行参数，脚本和 Agent 组件都从这里拿同一份配置。"""

    project_root: Path = PROJECT_ROOT
    dataset_root: Path = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload"
    raw_root: Path = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload" / "raw"
    questions_root: Path = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload" / "questions" / "group_a"
    processed_dir: Path = PROJECT_ROOT / "processed_data"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    index_dir: Path = PROJECT_ROOT / "processed_data" / "indexes"
    qwen_model: str = "qwen3.7-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_enable_thinking: bool = True
    max_evidence_chars: int = 6000
    top_k_retrieval: int = 30
    top_k_evidence: int = 8
    option_evidence_chars: int = 5000
    option_top_k_evidence: int = 6
    blind_top_docs: int = 8
    answer_max_tokens: int = 512
    answer_enable_thinking: bool = False
    option_judgement_max_tokens: int = 256
    option_judgement_enable_thinking: bool = False
    enable_multi_option_judgement: bool = True
    request_timeout_seconds: int = 120
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量和本地私有配置构造 Settings。"""
        return cls(
            dataset_root=_env_path("AFAC_DATASET_ROOT", cls.dataset_root),
            raw_root=_env_path("AFAC_RAW_ROOT", cls.raw_root),
            questions_root=_env_path("AFAC_QUESTIONS_ROOT", cls.questions_root),
            processed_dir=_env_path("AFAC_PROCESSED_DIR", cls.processed_dir),
            outputs_dir=_env_path("AFAC_OUTPUTS_DIR", cls.outputs_dir),
            index_dir=_env_path("AFAC_INDEX_DIR", cls.index_dir),
            qwen_model=_setting("AFAC_QWEN_MODEL", cls.qwen_model),
            qwen_base_url=_setting("AFAC_QWEN_BASE_URL", cls.qwen_base_url),
            qwen_enable_thinking=_env_bool("AFAC_QWEN_ENABLE_THINKING", cls.qwen_enable_thinking),
            max_evidence_chars=int(_setting("AFAC_MAX_EVIDENCE_CHARS", str(cls.max_evidence_chars))),
            top_k_retrieval=int(_setting("AFAC_TOP_K_RETRIEVAL", str(cls.top_k_retrieval))),
            top_k_evidence=int(_setting("AFAC_TOP_K_EVIDENCE", str(cls.top_k_evidence))),
            option_evidence_chars=int(_setting("AFAC_OPTION_EVIDENCE_CHARS", str(cls.option_evidence_chars))),
            option_top_k_evidence=int(_setting("AFAC_OPTION_TOP_K_EVIDENCE", str(cls.option_top_k_evidence))),
            blind_top_docs=int(_setting("AFAC_BLIND_TOP_DOCS", str(cls.blind_top_docs))),
            answer_max_tokens=int(_setting("AFAC_ANSWER_MAX_TOKENS", str(cls.answer_max_tokens))),
            answer_enable_thinking=_env_bool("AFAC_ANSWER_ENABLE_THINKING", cls.answer_enable_thinking),
            option_judgement_max_tokens=int(
                _setting("AFAC_OPTION_JUDGEMENT_MAX_TOKENS", str(cls.option_judgement_max_tokens))
            ),
            option_judgement_enable_thinking=_env_bool(
                "AFAC_OPTION_JUDGEMENT_ENABLE_THINKING", cls.option_judgement_enable_thinking
            ),
            enable_multi_option_judgement=_env_bool(
                "AFAC_ENABLE_MULTI_OPTION_JUDGEMENT", cls.enable_multi_option_judgement
            ),
        )

    def ensure_dirs(self) -> None:
        """确保运行产物目录存在。"""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


def get_api_key() -> str | None:
    """读取 Qwen/百炼兼容 API Key；只返回内存值，不写入任何文件。"""
    return (
        os.getenv("DASHSCOPE_API_KEY")
        or _local_value("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_API_KEY")
        or _local_value("BAILIAN_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or _local_value("QWEN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or _local_value("OPENAI_API_KEY")
    )
