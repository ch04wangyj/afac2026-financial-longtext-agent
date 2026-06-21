# AFAC2026 赛题四金融长文本 Agent

面向 AFAC2026 赛题四的金融长文本答题仓库。当前版本已经收敛为**无 embedding 的正式默认主线**，并保留两条 **LogicRAG 实验线** 作为受支持但非默认的运行路径。

## 当前版本提供什么

### 正式默认主线
- `doc_first_bm25f_expansion`
- 特征：词法检索、文档级 shortlist、chunk 级 sparse 检索、doc-first local expansion / aggregation、无 embedding
- 适用：当前默认提交 / smoke / sample / A100 正式运行

### 保留的 LogicRAG 实验线
- `logicrag_qwen_rrf`
  - Retrieval-first：先验证 LogicRAG 的 query planning / query bundle 对检索证据组织是否有帮助
- `logicrag_agent`
  - Full-agent：在 retrieval 之上启用更完整的 LogicRAG 推理执行链

### 不再作为当前维护入口的历史面
以下内容已从当前主 README 运行面移除，不应再被视为当前正式工作流：
- 已删除脚本：
  - `scripts/05_compare_rag.py`
  - `scripts/09_compare_runs.py`
  - `scripts/11_probe_retrieval_system.py`
  - `scripts/diagnose_answer_delta.py`
- 已退出 active runtime surface 的历史策略：
  - `question_options`
  - `rule_multi_rrf`
  - `field_boosted_rrf`
  - `logic_lite_rrf`
  - `linear_entity_rrf`
  - `graph_lite_rrf`
  - `crag_lite`

---

## 仓库结构速览

- `agent/`：核心代码
  - `preprocess/`：文档解析、Docling 适配、领域规则、索引字段构造
  - `index/`：BM25 / document index
  - `retrieve/`：默认主线、LogicRAG retrieval、claim/context/coverage 等检索组件
  - `reasoning/`：solver、LogicRAG、option-level / claim-centric / verifier 等推理链
  - `runtime/`：运行时配置加载、策略 contract、并发/guard、mode override
  - `io/`：JSONL 与输出目录辅助
- `scripts/`：用户入口脚本
- `configs/logicrag_runtime.yaml`：LogicRAG / thinking / 并发预算配置
- `docs/`：当前维护的辅助文档
- `tests/`：回归测试
- `theory/`：历史研究、路线图、参考资料和笔记（次级阅读，不是顶层运行手册）
- `VERSION_SCORE_LOG.md`：版本得分记录

---

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

配置 `DASHSCOPE_API_KEY` 后可真实调用 Qwen；未配置时可以用 `--dry-run` 验证链路。

私有配置可写入：
- `.env`
- 系统环境变量
- `agent/local_settings.py`（由 `agent/local_settings.example.py` 复制）

这些文件都不应提交到 git。

---

## 模型与运行时配置

默认模型：

```powershell
AFAC_QWEN_MODEL=qwen3.7-plus
```

高预算切换：

```powershell
AFAC_QWEN_MODEL=qwen3.7-max
```

LogicRAG / thinking / 并发预算配置集中在：
- `configs/logicrag_runtime.yaml`

当前配置里已经区分多种 thinking profile，例如：
- `logicrag_planner`
- `logicrag_query_bundle`
- `logicrag_refinement`
- `logicrag_rank_summary`
- `logicrag_final_compose`
- `multi_option_fallback`
- `claim_set_verification`

如果你要调整预算或开关，优先改这个 YAML，而不是把行为散落进脚本参数。

---

## 当前受支持的入口脚本

### 主运行脚本
- `scripts/01_prepare_docs.py`：预处理原始文档
- `scripts/02_build_index.py`：构建索引产物
- `scripts/03_run_questions.py`：跑默认主线或指定 runtime strategy
- `scripts/04_make_submission.py`：从 `answer_results.jsonl` 生成提交产物
- `scripts/06_smoke_by_domain.py`：按领域做 smoke
- `scripts/07_run_sample.py`：按 sampleN 运行样本集
- `scripts/08_report_results.py`：汇总 token / evidence / format 风险

### 维护与辅助脚本
- `scripts/10_cleanup_outputs.py`：安全清理历史输出
- `scripts/11_export_docling_samples.py`：导出 Docling 样本
- `scripts/12_analyze_docling_samples.py`：分析 Docling 样本并生成规则草案
- `scripts/12_refresh_extra_index_fields_from_existing_index.py`：基于现有索引重写 `chunks.jsonl` 的 `extra_index_fields`

---

## 默认主线工作流

### 1) 预处理 + 建索引 + dry-run

```powershell
python scripts\01_prepare_docs.py
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run
python scripts\04_make_submission.py
```

### 2) 小规模烟测

```powershell
python scripts\01_prepare_docs.py --limit 5
python scripts\02_build_index.py
python scripts\03_run_questions.py --dry-run --limit 5
python scripts\04_make_submission.py
```

### 3) 按领域 smoke

```powershell
python scripts\06_smoke_by_domain.py --dry-run --per-domain 1
python scripts\06_smoke_by_domain.py --per-domain 1
```

### 4) 分层样本验证

```powershell
python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260609
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609
python scripts\04_make_submission.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
python scripts\08_report_results.py --results outputs\samples\sample20\live\<timestamp>_<strategy>\answer_results.jsonl
```

### 5) Resume

```powershell
python scripts\03_run_questions.py --resume
python scripts\06_smoke_by_domain.py --per-domain 1 --resume
python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260609 --resume
```

---

## A 榜质量模式（V10 candidate）

`--a-board-quality` 在 V9 claim-centric 路径上增加三层确定性约束：

- **证据集合选择**：在字符预算内按文档、实体、数值、日期、条款后果和表格结构覆盖选择证据，抑制近重复 chunk。
- **指标规范化检索**：选项实体优先，使用财报披露名词典和 weighted RRF 对跨公司比较端点做逐文档召回。
- **claim 校准与集合级复核**：验证证据编号；对多选、证据不足、全称/复合/数值断言执行一次 exact-match 集合复核。
- **数值事实账本**：确定性抽取指标、年份、原值、单位和规范化值，供最终复核核对；不执行模型生成代码。

```powershell
python scripts\07_run_sample.py --dry-run --a-board-quality --sample-size 20 --per-domain 4 --seed 20260621
python scripts\07_run_sample.py --a-board-quality --sample-size 20 --per-domain 4 --seed 20260621
python scripts\03_run_questions.py --a-board-quality
```

该模式仍严格限制在题目给定的 `doc_ids` 内。新增组件只在质量模式启用，默认 `doc_first_bm25f_expansion` 主线行为不变。实验依据和边界见 `theory/references/notes/2026-06-21_v10-set-verification-and-fact-ledger.md`。

---

## LogicRAG 实验工作流

> **注意：LogicRAG 仍是保留实验线，不是默认正式主线。**

### Retrieval-first：`logicrag_qwen_rrf`

适合先看 LogicRAG 的 query planning / retrieval organization 是否值得继续推进，而不直接引入 full-agent 成本。

```powershell
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\07_run_sample.py --dry-run --sample-size 20 --per-domain 4 --seed 20260616
```

### Full-agent：`logicrag_agent`

```powershell
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\06_smoke_by_domain.py --dry-run --per-domain 1
AFAC_LOGICRAG_ENABLED=true AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\07_run_sample.py --sample-size 20 --per-domain 4 --seed 20260616
```

### 推荐顺序

1. 先跑默认主线 `doc_first_bm25f_expansion`
2. 再跑 `logicrag_qwen_rrf`
3. 只有在需要时再跑 `logicrag_agent`

---

## 输出目录与产物

当前输出目录规则见：
- `docs/output-layout.md`

最常见的目录类型：
- `outputs/tests/...`：`03` 的 limit/domain 子集测试输出，或其他 test-scope 运行
- `outputs/tests/smoke/...`：`06` smoke 输出
- `outputs/samples/sampleN/...`：`07` 样本运行输出
- `outputs/a100/full100/live/...`：A100 正式输出
- `outputs/docling_samples/...`：Docling 样本导出与分析输出

默认会与 `answer_results.jsonl` 同目录落盘的产物：
- `answer_results.jsonl`
- `answer.csv`
- `evidence.json`
- `token_usage.json`
- `run_report.md`
- `run_report.json`

---

## Docling 与预处理辅助工具

当前分支已经包含 Docling 相关辅助能力，但它们是**辅助开发/预处理工具**，不是顶层正式答题主流程。

### 导出样本

```powershell
python scripts\11_export_docling_samples.py --per-domain 1
```

输出目录：
- `outputs/docling_samples/<domain>/<doc_id>/...`

### 分析样本

```powershell
python scripts\12_analyze_docling_samples.py
```

输出目录：
- `outputs/docling_samples/analysis_summary.json`
- 每个样本目录下的 `analysis.md`

### 刷新 `extra_index_fields`

```powershell
python scripts\12_refresh_extra_index_fields_from_existing_index.py
```

这个脚本会根据当前索引内容重写：
- `processed_data/chunks.jsonl`

仅在你明确知道自己为什么要重写 `chunks.jsonl` 时使用。

---

## 版本记录与次级文档

### 版本得分记录
- `VERSION_SCORE_LOG.md`

### 当前维护的辅助文档
- `docs/output-layout.md`
- `docs/retrieval/nonembed_retrieval_notes.md`

### 历史 / 深度研究资料（次级阅读）
- `theory/`
- `theory/references/README.md`

这些文档可以帮助理解路线与取舍，但**不应替代本 README 的当前运行口径**。

---

## Development Rules

### Retrieval strategy override

不要改代码默认值来切换策略，优先通过环境变量覆盖：

```powershell
AFAC_RETRIEVAL_STRATEGY=doc_first_bm25f_expansion python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_agent python scripts\03_run_questions.py --dry-run --limit 2
```

运行时建议检查日志中的：
- `retrieval_strategy=...`
- `index=...`
- `doc_index_loaded=True/False`

### 其他约束
- 不使用 embedding、向量数据库或非 Qwen reranker
- `.env`、`processed_data/`、`outputs/` 不提交
- 所有 Qwen 调用必须记录 usage
- `03/06/07` 支持逐题 checkpoint 和 `--resume`
- GitHub 提交应使用 GitHub 允许的 noreply 身份

---

## 合并到 main 前的最小验证集

### 1) Runtime contract

```powershell
python -m pytest tests/test_runtime_strategy_contract.py -v
```

### 2) 默认主线策略

```powershell
python -m pytest tests/retrieval_system/test_default_retrieval_strategy.py -v
```

### 3) LogicRAG retrieval / solver

```powershell
python -m pytest tests/test_logicrag_retrieval.py tests/test_logicrag_solver.py -v
```

### 4) 输出目录逻辑

```powershell
python -m pytest tests/test_output_layout.py -v
```

### 5) 推荐 smoke

```powershell
python scripts\03_run_questions.py --dry-run --limit 2
AFAC_RETRIEVAL_STRATEGY=logicrag_qwen_rrf python scripts\07_run_sample.py --dry-run --sample-size 5 --per-domain 1 --seed 20260621
python scripts\04_make_submission.py --results <generated-answer_results.jsonl>
```

### 成功信号
- 默认主线日志里出现：`retrieval_strategy=doc_first_bm25f_expansion`
- 运行脚本输出：`wrote ... answer_results.jsonl`
- 提交脚本输出：`wrote submission artifacts to ...`

---

## 准备推到 main 的检查清单

在把当前分支合并到 `main` 之前，至少确认以下几点：

- [ ] `README.md` 与当前保留主线/实验线一致
- [ ] README 和顶层 docs 不再引用已删除脚本
- [ ] 默认主线 / LogicRAG 的最小验证集通过
- [ ] smoke 命令能真实跑通
- [ ] 输出目录说明与 `docs/output-layout.md` 一致
- [ ] 版本/分支说明足以让 reviewer 识别 **Keep / Removed / Experimental** 边界
- [ ] 提交作者身份使用 GitHub 允许的 noreply email

推荐合并方式：
1. 从 `ARS` 开 PR 到 `main`
2. 在 PR 描述中明确：保留主线、保留 LogicRAG、移除历史 compare/probe 面
3. 附上最小验证命令与结果
4. 通过 review 后 squash merge 到 `main`

---

## 给 reviewer 的建议 PR 摘要模板

```md
## Summary
- Refresh README for the retained default mainline and LogicRAG experiment lines
- Remove stale top-level references to deleted compare/probe workflows
- Align output-layout and release-readiness instructions with the current branch

## Kept
- `doc_first_bm25f_expansion`
- `logicrag_qwen_rrf`
- `logicrag_agent`

## Removed from active docs surface
- deleted compare/probe scripts
- legacy retrieval variants as active runtime entrypoints

## Validation
- runtime strategy contract tests
- default retrieval strategy test
- LogicRAG retrieval / solver tests
- output layout tests
- dry-run smoke + submission artifact generation
```

---

## Historical Note

当前仓库仍保留一定数量的历史研究文档、理论路线图和内部参考资料；这些内容用于复盘与研究，不代表当前默认运行面。若 README 与 `theory/` 某些历史文档描述冲突，以 README 当前版本和受支持脚本为准。
