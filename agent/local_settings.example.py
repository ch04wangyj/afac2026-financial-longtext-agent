"""本地私有配置示例。

如需本机直写密钥，可复制为 agent/local_settings.py；真实文件已被 .gitignore 排除。
不要在本示例文件或任何会提交到 git 的文件里写真实 API Key。
"""

DASHSCOPE_API_KEY = "replace-with-your-key"

# qwen3.7-plus 适合默认跑题；qwen3.7-max 适合难题或冲榜复核。
AFAC_QWEN_MODEL = "qwen3.7-plus"
# AFAC_QWEN_MODEL = "qwen3.7-max"

AFAC_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AFAC_QWEN_ENABLE_THINKING = True  # 仅作为未显式指定 profile 时的兜底开关
AFAC_ANSWER_MAX_TOKENS = 384  # 推荐与 configs/logicrag_runtime.yaml 中 answer_single_pass 保持一致
AFAC_ANSWER_ENABLE_THINKING = False  # 兼容旧路径；主控制已迁移到 thinking_profiles
AFAC_OPTION_EVIDENCE_CHARS = 5000
AFAC_OPTION_TOP_K_EVIDENCE = 6
AFAC_OPTION_JUDGEMENT_MAX_TOKENS = 192  # 推荐与 option_judgement profile 保持一致
AFAC_OPTION_JUDGEMENT_ENABLE_THINKING = False  # 兼容旧路径；主控制已迁移到 thinking_profiles
AFAC_ENABLE_MULTI_OPTION_JUDGEMENT = True
AFAC_BLIND_TOP_DOCS = 8
