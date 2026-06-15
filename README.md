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

分层抽样 20 题并生成诊断报告：

```powershell
python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260609
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609
python scripts\04_make_submission.py
python scripts\08_report_results.py
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

已完成的全量横向评估、框架路线图和代码 review 见：

```text
theory/AFAC2026_赛题四_规则核对与偏差修正.md
theory/AFAC2026_赛题四_RAG横向评估与代码Review.md
theory/AFAC2026_赛题四_RAG框架扩展路线图.md
```

## Outputs

- `processed_data/documents.jsonl`: 解析后的文档。
- `processed_data/chunks.jsonl`: 分块后的证据块。
- `processed_data/indexes/bm25_index.pkl`: 词法索引。
- `processed_data/indexes/document_bm25_index.pkl`: B 榜盲搜用文档级词法索引。
- `outputs/answer_results.jsonl`: 每题答案、证据和 Token。
- `outputs/answer.csv`: 比赛提交文件。
- `outputs/evidence.json`: 可审计证据。
- `outputs/token_usage.json`: 汇总 Token。
- `outputs/rag_compare/`: 不同 RAG/tokenizer 方案的检索命中横向比较。
- `outputs/run_report.md`: 抽样运行后的 Token、证据覆盖和格式风险诊断。

## Current Baseline

截至 2026-06-09，已完成同一 20 题分层样本的真实 Qwen 回归：

| run | total_tokens | answer_changes_vs_full | issues |
|---|---:|---:|---|
| full evidence 8/6000 | 158216 | 0 | 无格式/证据/verdict 问题 |
| adaptive evidence | 145895 | 0 | 无格式/证据/verdict 问题 |

自适应证据预算在样本上比 full evidence 少 `12321` tokens，且答案完全一致。完整报告在：

```text
outputs/live_sample20_adaptive/run_report.md
outputs/compare_sample20_full_vs_adaptive/comparison.md
```

已完成 A 组 100 题真实运行：

| output | value |
|---|---:|
| question_rows | 100 |
| prompt_tokens | 989706 |
| completion_tokens | 12586 |
| total_tokens | 1002292 |
| detected issues | 0 |

全量产物在：

```text
outputs/a_full_adaptive/answer.csv
outputs/a_full_adaptive/evidence.json
outputs/a_full_adaptive/token_usage.json
outputs/a_full_adaptive/run_report.md
```

## Development Rules

- 不使用 embedding、向量数据库或非 Qwen reranker。
- `.env`、`processed_data/`、`outputs/` 不提交。
- 所有 Qwen 调用必须记录 usage。
- `03/06/07` 运行脚本已支持逐题 checkpoint 和 `--resume`，全量 A 组真实调用建议始终指定独立 `AFAC_OUTPUTS_DIR`。
- 默认模型为 `qwen3.7-plus`，可切换到 `qwen3.7-max`；使用百炼 OpenAI-compatible API 和流式输出，并尽量通过 `stream_options.include_usage` 获取真实 Token。
- 为控制 Token，普通答题和逐选项判断默认覆盖为 `enable_thinking=false`；低置信复核时再显式打开 thinking 或切换 max。
- 逐选项判断使用独立证据预算：默认 `AFAC_OPTION_TOP_K_EVIDENCE=6`、`AFAC_OPTION_EVIDENCE_CHARS=5000`；研报自动提升到 full 预算，避免遗漏长文档趋势证据。
- 原版 GraphRAG/LightRAG/HippoRAG/RAPTOR/Self-RAG/OpenSPG 不进入默认链路；只允许无 embedding、无非 Qwen 模型的 lite 思想作为实验分支。
- V1 默认 `question_options` 主检索；多选题已启用逐选项判断；B 榜无 `doc_ids` 时使用文档级 BM25 盲搜候选。
- V2 后续再加入 Qwen 子问题 DAG、IterKey、Calculator、IRCoT-lite 和低置信 CoVe。
