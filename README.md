# AFAC2026 赛题四金融长文本 Agent

默认使用无 embedding 的稀疏检索主线：`doc_first_bm25f_expansion`。

当前仓库只保留两类受支持路径：

1. **正式默认主线**：`doc_first_bm25f_expansion`
2. **保留的 LogicRAG 实验线**：`logicrag_qwen_rrf`、`logicrag_agent`

其余早期 sparse baseline、横向对照变体和 compare/probe 脚本已不再作为当前维护入口。

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

配置 `DASHSCOPE_API_KEY` 后可真实调用 Qwen。未配置时可使用 `--dry-run` 验证链路。

本地私有配置也可写入：
- `.env`
- 系统环境变量
- `agent/local_settings.py`（由 `agent/local_settings.example.py` 复制而来）

这些私有配置均已被 `.gitignore` 排除。

## 模型切换

```powershell
# 默认
AFAC_QWEN_MODEL=qwen3.7-plus

# 更高预算
AFAC_QWEN_MODEL=qwen3.7-max
```

## Default Mainline Pipeline

```powershell
python scripts\01_prepare_docs.py
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run
python scripts\04_make_submission.py
```

## 常用烟测

```powershell
python scripts\01_prepare_docs.py --limit 5
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run --limit 5
python scripts\04_make_submission.py
```

## 分层抽样验证

```powershell
python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260609
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609
python scripts\04_make_submission.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
python scripts\08_report_results.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
```

## Resume

```powershell
python scripts\03_run_questions.py --resume
python scripts\06_smoke_by_domain.py --per-domain 1 --resume
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609 --resume
```

## LogicRAG Experimental Paths

> **注意：LogicRAG 仍是实验线，不是默认正式主线。**

### 1) Retrieval-first：`logicrag_qwen_rrf`

适合先验证 LogicRAG 的 query planning 是否能带来更好的检索证据组织，而不直接引入 full-agent 成本。

```powershell
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260616
```

### 2) Full-agent：`logicrag_agent`

```powershell
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\06_smoke_by_domain.py --dry-run --per-domain 1
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260616
```

### 建议顺序

1. 先跑默认主线 `doc_first_bm25f_expansion`
2. 再跑 `logicrag_qwen_rrf`
3. 最后在需要时跑 `logicrag_agent`

## Outputs Layout

- `processed_data/documents.jsonl`: 解析后的文档
- `processed_data/chunks.jsonl`: 分块后的证据块
- `processed_data/indexes/bm25_index.pkl`: chunk 级词法索引
- `processed_data/indexes/document_bm25_index.pkl`: 文档级词法索引
- `outputs/tests/smoke/...`: smoke / dry-run / 小规模验证
- `outputs/samples/sample20/...`: sample20 评估输出
- `outputs/samples/sample40/...`: 其他 sample 规模输出
- `outputs/a100/full100/live/...`: A 榜 100 题正式运行输出

提交与诊断产物会默认贴靠 `answer_results.jsonl` 所在目录共同落盘：

- `answer_results.jsonl`
- `answer.csv`
- `evidence.json`
- `token_usage.json`
- `run_report.md`
- `run_report.json`

完整规范见：`docs/output-layout.md`

## Development Rules

### Retrieval strategy override

如果需要临时切换路径，不修改代码默认值，优先通过环境变量覆盖：

```powershell
AFAC_RETRIEVAL_STRATEGY=doc_first_bm25f_expansion python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\03_run_questions.py --dry-run --limit 2
```

运行前后建议检查日志中的：
- `retrieval_strategy=...`
- `index=...`
- `doc_index_loaded=True/False`

## Operational Notes

- 不使用 embedding、向量数据库或非 Qwen reranker
- `.env`、`processed_data/`、`outputs/` 不提交
- 所有 Qwen 调用必须记录 usage
- `03/06/07` 运行脚本支持逐题 checkpoint 和 `--resume`
- 如需清理历史无用输出，先执行：

```powershell
python scripts\10_cleanup_outputs.py --dry-run
```

确认后再执行：

```powershell
python scripts\10_cleanup_outputs.py --apply
```

- 默认模型为 `qwen3.7-plus`，可切换到 `qwen3.7-max`
- LogicRAG 的 thinking 预算配置在 `configs/logicrag_runtime.yaml`
- 默认主线仍是 `doc_first_bm25f_expansion`；LogicRAG 为保留实验线
