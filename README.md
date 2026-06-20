# AFAC2026 赛题四金融长文本 Agent

V1 稳健检索版工程骨架。默认不使用 embedding、不加载非 Qwen 压缩模型，采用规则抽取式压缩和 BM25 词法检索。

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

配置 `DASHSCOPE_API_KEY` 后可真实调用 Qwen。未配置时可用 `--dry-run` 验证链路。真实 API Key 可以写入 `.env`、系统环境变量，或复制 `agent/local_settings.example.py` 为 `agent/local_settings.py` 后本地直写；`.env` 和 `agent/local_settings.py` 都已被 `.gitignore` 排除，不能提交到 git。

模型可在本地配置中切换：

```powershell
# 默认
AFAC_QWEN_MODEL=qwen3.7-plus

# 难题/冲榜可切换
AFAC_QWEN_MODEL=qwen3.7-max
```

## Pipeline

```powershell
python scripts\01_prepare_docs.py
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run
python scripts\04_make_submission.py
```

常用烟测：

```powershell
python scripts\01_prepare_docs.py --limit 5
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run --limit 5
python scripts\04_make_submission.py
```

按五个领域各抽 1 题做 smoke：

```powershell
python scripts\06_smoke_by_domain.py --dry-run --per-domain 1
python scripts\06_smoke_by_domain.py --per-domain 1
```

默认输出会进入：

```text
outputs/tests/smoke/dry/<timestamp>_<strategy>/
outputs/tests/smoke/live/<timestamp>_<strategy>/
```

分层抽样 20 题并生成诊断报告：

```powershell
python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260609
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609
python scripts\04_make_submission.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
python scripts\08_report_results.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
```

默认输出会进入：

```text
outputs/samples/sample20/dry/<timestamp>_<strategy>/
outputs/samples/sample20/live/<timestamp>_<strategy>/
```

长任务中断后可续跑：

```powershell
python scripts\03_run_questions.py --resume
python scripts\06_smoke_by_domain.py --per-domain 1 --resume
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609 --resume
```

横向比较不同 RAG 检索方式，先不考虑 Token：

```powershell
python scripts\05_compare_rag.py --domains financial_contracts --limit 5
```

输出在 `outputs/rag_compare/`，使用 A 榜 `doc_ids` 作为检索命中代理指标。`recall_at_10` 和 `all_gold_at_10` 越高，后续 Qwen 推理越可能拿到正确证据；这不是最终答案准确率。

比较 Graph/Logic/Linear/CRAG lite variants：

```powershell
python scripts\05_compare_rag.py --tokenizer-modes mixed --variants question_options rule_multi_rrf field_boosted_rrf logic_lite_rrf linear_entity_rrf graph_lite_rrf crag_lite --output-name rag_compare_frameworks_mixed
```

## LogicRAG Experimental Modes

> **注意：LogicRAG 目前仍是实验开关，不是默认主线。**
> 当前正式默认检索主线已经切换为 `doc_first_bm25f_expansion`：即 **BM25F-lite + doc-first local expansion/aggregation + offline expansion-field-enhanced index**。`logic_lite_rrf` / `crag_lite` / `logicrag_qwen_rrf` / `logicrag_agent` 仍作为实验或对照分支存在，而不是默认正式线上链路。

### 1) Retrieval-only：`logicrag_qwen_rrf`

适合先看检索代理指标是否值得继续，不直接引入 full-agent 的额外 Token 成本。

```powershell
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260616
python scripts\05_compare_rag.py --tokenizer-modes mixed --variants question_options logic_lite_rrf logicrag_qwen_rrf --limit 40 --output-name rag_compare_logicrag_qwen_sample40
```

常见输出：

```text
outputs/rag_compare_logicrag_qwen_sample40/report.md
outputs/rag_compare_logicrag_qwen_finance/report.md
```

### 2) Full-agent：`logicrag_agent`

只有在 retrieval-only 已证明值得继续时，才建议打开 full-agent 试验。

```powershell
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\06_smoke_by_domain.py --dry-run --per-domain 1
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260616
python scripts\09_compare_runs.py --baseline outputs\samples\sample20\live\<baseline_stamp>_<baseline_strategy>\answer_results.jsonl --candidate outputs\samples\sample20\live\<candidate_stamp>_<candidate_strategy>\answer_results.jsonl
```

常见输出：

```text
outputs/tests/smoke/dry/<timestamp>_logicrag_agent/answer_results.jsonl
outputs/samples/sample20/live/<timestamp>_logicrag_agent/run_report.md
outputs/samples/compare/<baseline>__vs__<candidate>/comparison.md
```

### 3) 推荐使用顺序

1. 先跑正式默认链路 `doc_first_bm25f_expansion`，建立 baseline。
2. 再跑 `logicrag_qwen_rrf` 看 retrieval proxy 是否优于实验对照（如 `logic_lite_rrf`）。
3. 只有 retrieval-only 结果达标，再打开 `logicrag_agent` 做 smoke / sample20。
4. 如果 Token 明显上升但 retrieval 或答案质量没有稳定改善，则保持 LogicRAG 为实验分支，不升级为默认主线。

已完成的全量横向评估、框架路线图和代码 review 见：

```text
theory/AFAC2026_赛题四_规则核对与偏差修正.md
theory/AFAC2026_赛题四_RAG横向评估与代码Review.md
theory/AFAC2026_赛题四_RAG框架扩展路线图.md
```

## Outputs Layout

- `processed_data/documents.jsonl`: 解析后的文档。
- `processed_data/chunks.jsonl`: 分块后的证据块。
- `processed_data/indexes/bm25_index.pkl`: 词法索引。
- `processed_data/indexes/document_bm25_index.pkl`: B 榜盲搜用文档级词法索引。
- `outputs/tests/smoke/...`: smoke / dry-run / 小规模验证输出。
- `outputs/tests/retrieval_compare/...`: `scripts/05_compare_rag.py` 的检索对比输出。
- `outputs/samples/sample20/...`: sample20 评估输出。
- `outputs/samples/sample40/...`: 其他 sample 规模输出。
- `outputs/samples/compare/...`: 样本运行之间的对比报告。
- `outputs/a100/full100/live/...`: A 榜 100 题正式运行输出。
- `outputs/a100/compare/...`: A100 运行对比报告。

提交与诊断产物会默认贴靠 `answer_results.jsonl` 所在目录共同落盘：

- `answer_results.jsonl`
- `answer.csv`
- `evidence.json`
- `token_usage.json`
- `run_report.md`
- `run_report.json`

完整规范见：`docs/output-layout.md`

## Historical Note (Not a Valid Current Baseline)

以下 `full evidence / adaptive evidence` 结果仅保留为**历史实验记录**，**不再作为当前 baseline，也不应作为可提交方案依据**。原因是这条旧方案不属于当前 ARS/LogicRAG 评估口径，继续拿它做比较会混淆“当前无 embedding 主线内部”的真实收益判断。

截至 2026-06-09，曾完成同一 20 题分层样本的真实 Qwen 回归：

| run | total_tokens | answer_changes_vs_full | issues |
|---|---:|---:|---|
| full evidence 8/6000 | 158216 | 0 | 无格式/证据/verdict 问题 |
| adaptive evidence | 145895 | 0 | 无格式/证据/verdict 问题 |

对应历史产物仍可留档，但后续 stop/go 判断不再基于它们：

```text
outputs/live_sample20_adaptive/run_report.md
outputs/compare_sample20_full_vs_adaptive/comparison.md
outputs/a_full_adaptive/answer.csv
outputs/a_full_adaptive/evidence.json
outputs/a_full_adaptive/token_usage.json
outputs/a_full_adaptive/run_report.md
```

## Current Valid Baseline Scope

当前允许作为正式比较对象的基线范围应区分“正式默认主线”和“实验/对照分支”：

- 正式默认主线：`doc_first_bm25f_expansion`
- 实验/对照分支：`logic_lite_rrf`
- 实验/对照分支：`crag_lite`
- 传统稀疏对照：`question_options`

其中：

- `logicrag_qwen_rrf` / `logicrag_agent` 只作为 ARS / LogicRAG 实验分支
- 不再把旧 adaptive/full evidence 方案当作默认主线或 rollout 依据
- `question_options` 继续保留为 legacy sparse baseline，但不再是正式默认线上链路

## Development Rules

### Retrieval strategy rollback

如果需要临时回滚默认检索策略，不要改代码默认值，优先通过环境变量覆盖：

```powershell
AFAC_RETRIEVAL_STRATEGY=hybrid python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=bm25f_lite_rrf python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\03_run_questions.py --dry-run --limit 2
```

回滚前后都应检查启动日志中的：
- `retrieval_strategy=...`
- `index=...`
- `doc_index_loaded=True/False`

如果部署环境没有同步新的 `processed_data/chunks.jsonl`、`bm25_index.pkl`、`document_bm25_index.pkl`，优先补齐索引产物，再决定是否真正回滚策略。

- 不使用 embedding、向量数据库或非 Qwen reranker。
- `.env`、`processed_data/`、`outputs/` 不提交。
- 所有 Qwen 调用必须记录 usage。
- `03/06/07` 运行脚本已支持逐题 checkpoint 和 `--resume`，全量 A 组真实调用建议始终指定独立 `AFAC_OUTPUTS_DIR`。
- 如需清理历史无用输出，先执行 `python scripts/10_cleanup_outputs.py --dry-run`，确认后再执行 `--apply`。
- 默认模型为 `qwen3.7-plus`，可切换到 `qwen3.7-max`；使用百炼 OpenAI-compatible API 和流式输出，并尽量通过 `stream_options.include_usage` 获取真实 Token。
- 思考预算按步骤分层配置在 `configs/logicrag_runtime.yaml`：`logicrag_planner` / `logicrag_final_compose` 默认高预算，`logicrag_rank_summary` 中预算，普通答题与逐选项判断默认低预算或关闭 thinking；仅在 `multi_option_fallback` 这类复核步骤再回升预算。
- 逐选项判断使用独立证据预算：默认 `AFAC_OPTION_TOP_K_EVIDENCE=6`、`AFAC_OPTION_EVIDENCE_CHARS=5000`；研报自动提升到 full 预算，避免遗漏长文档趋势证据。
- 原版 GraphRAG/LightRAG/HippoRAG/RAPTOR/Self-RAG/OpenSPG 不进入默认链路；只允许无 embedding、无非 Qwen 模型的 lite 思想作为实验分支。
- V1 默认 `doc_first_bm25f_expansion` 主检索：BM25F-lite + doc-first local expansion/aggregation + offline expansion-field-enhanced index；多选题已启用逐选项判断；B 榜无 `doc_ids` 时使用文档级 BM25 盲搜候选。
- 如需临时回退旧路径，只通过 `AFAC_RETRIEVAL_STRATEGY` 显式指定（如 `hybrid`、`bm25f_lite_rrf`、`logicrag_agent`），不修改代码默认值。

