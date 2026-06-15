"""本地私有配置示例。

如需本机直写密钥，可复制为 agent/local_settings.py；真实文件已被 .gitignore 排除。
不要在本示例文件或任何会提交到 git 的文件里写真实 API Key。
"""

DASHSCOPE_API_KEY = "replace-with-your-key"

# qwen3.7-plus 适合默认跑题；qwen3.7-max 适合难题或冲榜复核。
AFAC_QWEN_MODEL = "qwen3.7-plus"
# AFAC_QWEN_MODEL = "qwen3.7-max"

AFAC_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
AFAC_QWEN_ENABLE_THINKING = True
AFAC_QWEN_STREAM = True
AFAC_QWEN_STREAM_INCLUDE_USAGE = True
AFAC_ANSWER_MAX_TOKENS = 512
AFAC_ANSWER_ENABLE_THINKING = False
AFAC_OPTION_EVIDENCE_CHARS = 5000
AFAC_OPTION_TOP_K_EVIDENCE = 6
AFAC_OPTION_JUDGEMENT_MAX_TOKENS = 256
AFAC_OPTION_JUDGEMENT_ENABLE_THINKING = False
AFAC_ENABLE_MULTI_OPTION_JUDGEMENT = True
AFAC_BLIND_TOP_DOCS = 8
