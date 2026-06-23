# AFAC2026 赛题四金融长文本 Agent

面向 AFAC2026 赛题四的金融长文本答题仓库。当前代码为 **V14 官方版本**：在 V13 原子谓词检索之上增加确定性 PDF 坐标版面解析、矢量线/无线框表格恢复、表头与单位绑定，并继续满足 Qwen-only、无 embedding 的赛题约束。

| 状态 | 当前值 |
|---|---|
| 官方最佳版本 | V14 deterministic PDF layout + conservative merge |
| 官方最佳得分 | **68.69** |
| 反推准确率 | 按本地留存 Token `312,541` 反推约 **70 / 100**（仅供参考） |
| V13 官方结果 | **66.2592**，Token `377,650` |
| V12 官方结果 | **57.7848**，反推准确率 **67 / 100**，Token `2,292,333` |
| 当前可提交 A100 | 100 题，`total_tokens=312,541` |

## 当前版本提供什么

### 正式默认主线
- `v14_layout_precise`
- 特征：V13 原子子块 + PyMuPDF 坐标块 + 表格行、独立父块仓库、BM25F、谓词真实值查询、逐文档支持/反证选择、无 embedding
- 适用：当前 A100 正式运行与提交；`doc_first_bm25f_expansion` 保留为历史稳定基线

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
- `scripts/04_make_submission.py`：合并基础/覆盖结果，校验后生成提交产物
- `scripts/06_smoke_by_domain.py`：按领域做 smoke
- `scripts/07_run_sample.py`：按 sampleN 运行样本集
- `scripts/08_report_results.py`：汇总 token / evidence / format 风险
- `scripts/09_eval_answer_devset.py`：按人工核验 dev set 计算答案 exact-match 与关键证据召回

### 维护与辅助脚本
- `scripts/10_cleanup_outputs.py`：安全清理历史输出
- `scripts/11_export_docling_samples.py`：导出 Docling 样本
- `scripts/12_analyze_docling_samples.py`：分析 Docling 样本并生成规则草案
- `scripts/12_refresh_extra_index_fields_from_existing_index.py`：基于现有索引重写 `chunks.jsonl` 的 `extra_index_fields`
- `scripts/13_augment_financial_metric_rows.py`：从现有财报 text chunks 生成指标行，无需重新解析 PDF
- `scripts/14_run_exhaustive_verifier.py`：运行 V12 文档级高召回裁决与可选反证审计
- `scripts/15_build_hierarchical_index.py`：构建 V13 原子子块、父块仓库和层级 BM25F 索引
- `scripts/16_run_precise_verifier.py`：运行 V13 谓词真实值与支持/反证精确验证器
- `scripts/17_reconcile_results.py`：弱证据回退 V12，强变化执行独立 Qwen 审计
- `scripts/18_apply_review_overrides.py`：应用逐条原文复核结论且保持真实 Token 统计
- `scripts/19_build_layout_index.py`：构建 V14 确定性 PDF 版面/表格增量索引
- `scripts/20_merge_layout_candidate.py`：用 V14 结果替换受影响题 Token，并保守融合人工核验答案

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

## A 榜质量模式（V11 candidate）

`--a-board-quality` 在 V9 claim-centric 路径上增加三层确定性约束：

- **证据集合选择**：在字符预算内按文档、实体、数值、日期、条款后果和表格结构覆盖选择证据，抑制近重复 chunk。
- **指标规范化检索**：选项实体优先，使用财报披露名词典和 weighted RRF 对跨公司比较端点做逐文档召回。
- **claim 校准与集合级复核**：验证证据编号；对多选、证据不足、全称/复合/数值断言执行一次 exact-match 集合复核。
- **数值事实账本**：确定性抽取指标、年份、原值、单位和规范化值，供最终复核核对；不执行模型生成代码。
- **财务指标行索引**：将表格退化文本确定性转换为 `metric/year/value/unit/header/cells` 短 chunk，减少跨列错配。
- **受限计算 DSL**：仅允许 `compare/difference/ratio/growth_rate`，操作数必须绑定已抽取事实并通过单位检查。

```powershell
python scripts\07_run_sample.py --dry-run --a-board-quality --sample-size 20 --per-domain 4 --seed 20260621
python scripts\07_run_sample.py --a-board-quality --sample-size 20 --per-domain 4 --seed 20260621
python scripts\03_run_questions.py --a-board-quality
```

该模式仍严格限制在题目给定的 `doc_ids` 内。新增组件只在质量模式启用，默认 `doc_first_bm25f_expansion` 主线行为不变。实验依据和边界见：

- `theory/references/notes/2026-06-21_v10-set-verification-and-fact-ledger.md`
- `theory/references/notes/2026-06-21_v11-financial-row-index-and-dev-gate.md`

### 财报指标行实验索引

```powershell
python scripts\13_augment_financial_metric_rows.py --output processed_data\chunks_financial_rows.jsonl
python scripts\02_build_index.py --chunks processed_data\chunks_financial_rows.jsonl --index processed_data\index_financial_rows\bm25_index.pkl --skip-doc-index
$env:AFAC_INDEX_DIR="processed_data\index_financial_rows"
python scripts\07_run_sample.py --a-board-quality --qids fin_a_005 fin_a_011
```

### 答案级回归门禁

```powershell
python scripts\09_eval_answer_devset.py --results <answer_results_1.jsonl> <answer_results_2.jsonl> --strict
```

开发集位于 `devsets/answer_level_v1.jsonl`。人工标签必须绑定当前题面和数据版本；不能仅按 qid 复用官网旧示例答案。

### V11 历史提交

当前仓库中唯一完整 A100 底稿是 `outputs/a_full_adaptive/answer_results.jsonl`。以下命令只用 V11 已核验并发生答案变化的两道财报结果覆盖底稿，同时执行 100 题完整性、答案格式和 Token 一致性校验：

```powershell
python scripts\04_make_submission.py `
  --results outputs\a_full_adaptive\answer_results.jsonl `
  --override-results outputs\v11_financial_rows_live\answer_results.jsonl `
  --output-dir outputs\submissions\v11_candidate_20260621 `
  --require-complete
```

生成文件：`outputs/submissions/v11_candidate_20260621/answer.csv`。

| 校验项 | 结果 |
|---|---:|
| 题目覆盖 | 100 / 100 |
| 重复或额外 qid | 0 |
| `fin_a_005` | `ABD` |
| `fin_a_011` | `ACD` |
| `summary.total_tokens` | 1,030,141 |
| 文件编码 | UTF-8 with BOM |

`reg_a_014` 的 V11 回归答案与底稿同为 `ABD`，因此提交候选保留 Token 更低的底稿记录。`outputs/` 按仓库规则不提交 Git，正式 CSV 由上述命令本地复现。

### V12 历史提交

V12 不再让 BM25 Top-K 决定最终证据边界。每个选项都对题目给定的每份文档单独检索，并增加非年份数值/完整日期/实体精确扫描、同页和相邻块展开。首轮统一裁决后，只对与 V11 不一致的题执行反证审计。

运行入口：

```powershell
python scripts\14_run_exhaustive_verifier.py `
  --domains financial_reports `
  --index processed_data\index_financial_rows\bm25_index.pkl `
  --output-dir outputs\v12_financial_full_max `
  --model qwen3.7-max `
  --workers 4
```

对差异题增加 `--audit` 后，将各域结果按优先级合并：

```powershell
python scripts\04_make_submission.py `
  --results outputs\submissions\v11_candidate_20260621\answer_results.jsonl `
  --override-results outputs\v12_financial_full_max\answer_results.jsonl `
  --override-results outputs\v12_remaining80_max\answer_results.jsonl `
  --override-results outputs\v12_financial_patch_max\answer_results.jsonl `
  --override-results outputs\v12_fin013_focused_max\answer_results.jsonl `
  --override-results outputs\v12_changed42_audit_max\answer_results.jsonl `
  --output-dir outputs\submissions\v12_exhaustive_audit_20260622 `
  --require-complete
```

生成文件：`outputs/submissions/v12_exhaustive_audit_20260622/answer.csv`。

| 校验项 | 结果 |
|---|---:|
| 题目覆盖 | 100 / 100 |
| 相对 V11 答案变化 | 38 |
| 开发集 exact-match | 3 / 3 |
| `summary.total_tokens` | 2,292,333 |
| 官网得分 | 57.7848 |
| 反推准确率 | 67 / 100 |
| 文件编码 | UTF-8 with BOM |

V12 相对 V11 多答对 5 题，但额外约 126 万 Token 抵消了准确率收益。研究与实现说明见 `theory/references/notes/2026-06-22_v12-exhaustive-document-verifier.md`。

### V13 官方版本

V13 不再扩大 Top-K，而是先把旧页级块重建为原子子块。检索查询分成“候选值支持”和“不携带候选值的谓词真实值”，错误选项不会再仅凭虚假数值牵引检索。弱证据变化回退 V12，只有可复核变化才进入最终答案。

```powershell
python scripts\15_build_hierarchical_index.py

python scripts\16_run_precise_verifier.py `
  --output-dir outputs\v13_full_no_thinking `
  --workers 8 `
  --no-thinking

python scripts\17_reconcile_results.py `
  --results outputs\v13_full_no_thinking\answer_results.jsonl `
  --baseline outputs\submissions\v12_exhaustive_audit_20260622\answer_results.jsonl `
  --output outputs\v13_reconciled_thinking\answer_results.jsonl `
  --workers 6 `
  --thinking

python scripts\18_apply_review_overrides.py `
  --results outputs\v13_reconciled_thinking\answer_results.jsonl `
  --output outputs\v13_final_reviewed\answer_results.jsonl

python scripts\04_make_submission.py `
  --results outputs\v13_final_reviewed\answer_results.jsonl `
  --output-dir outputs\submissions\v13_precise_reviewed_20260622 `
  --require-complete
```

| 校验项 | 结果 |
|---|---:|
| 题目覆盖 | 100 / 100 |
| 相对 V12 最终变化 | 6 |
| 开发集 exact-match | 3 / 3 |
| `summary.total_tokens` | 377,650 |
| 官网得分 | **66.2592** |
| 文件编码 | UTF-8 with BOM |

提交文件：`outputs/submissions/v13_precise_reviewed_20260622/answer.csv`。官网未返回原始正确题数，且本地留存 CSV 的 Token 与实际提交口径存在差异，因此不根据得分强行反推准确率。V13 说明见 `theory/references/notes/2026-06-22_v13-atomic-predicate-verifier.md`。

### V14 官方版本

V14 不用新的视觉模型替换全文解析，而是在 V13 语料旁路增加确定性版面证据：读取 PDF 字符坐标和矢量线，恢复有框/无线框表格，将标题、单位、层级表头重复绑定到每个数据行。该设计不产生预处理 Token，也不引入赛题禁止的 embedding 或非 Qwen 模型。

```powershell
python scripts\19_build_layout_index.py --strict

python scripts\16_run_precise_verifier.py `
  --index processed_data\v14_layout\bm25_index.pkl `
  --output-dir outputs\v14_layout_40_final `
  --domains financial_reports research `
  --workers 8 `
  --no-thinking `
  --strategy-name v14_layout_precise

python scripts\20_merge_layout_candidate.py `
  --baseline outputs\v13_final_reviewed\answer_results.jsonl `
  --candidate outputs\v14_layout_40_final\answer_results.jsonl `
  --output outputs\v14_layout_final\answer_results.jsonl

python scripts\09_eval_answer_devset.py `
  --results outputs\v14_layout_final\answer_results.jsonl `
  --strict

python scripts\04_make_submission.py `
  --results outputs\v14_layout_final\answer_results.jsonl `
  --output-dir outputs\submissions\v14_layout_reviewed_20260622 `
  --require-complete
```

| 校验项 | 结果 |
|---|---:|
| 实际解析 PDF | 24 |
| 解析失败 | 0 |
| V13 原子子块 | 38,544 |
| V14 增量版面块 | 32,663 |
| 其中结构化表格行 | 14,302 |
| 合并索引子块 | 70,857 |
| 受影响 40 题 Token | 104,489 |
| A100 `summary.total_tokens` | **312,541** |
| 相对 V13 Token | **-17.2%** |
| 开发集 exact-match | 3 / 3 |
| 相对 V13 最终答案变化 | 2 |
| 官网得分 | **68.69** |
| 反推准确率（本地 Token） | 约 70 / 100 |
| 文件编码 | UTF-8 with BOM |

提交文件：`outputs/submissions/v14_layout_reviewed_20260622/answer.csv`。SHA-256：`DF843CDA104FD2E4409EBA8EDF79D9CC5903B2CCAEF37D0031C5FCD5874B7403`。V14 索引加载内存高于 V13，正式运行时不要同时启动多个 V14 进程。详细边界见 `theory/references/notes/2026-06-22_v14-deterministic-pdf-layout.md`。

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
- 提交脚本输出：`wrote 100 validated submission rows to ...`

---

## Git 协作检查清单

当前协作目标分支为 `main`。每次推送前至少确认：

- [ ] `git fetch origin` 后本地 `main` 未落后远端
- [ ] 未暂存 `.env`、`outputs/`、`processed_data/` 或真实 API Key
- [ ] 相关测试和 `python -m compileall agent scripts tests` 通过
- [ ] README、`VERSION_SCORE_LOG.md` 与运行时代码口径一致
- [ ] 正式 CSV 通过 `--require-complete` 校验
- [ ] 只有官网真实提交结果才能更新“当前最佳得分”

---

## Historical Note

当前仓库仍保留一定数量的历史研究文档、理论路线图和内部参考资料；这些内容用于复盘与研究，不代表当前默认运行面。若 README 与 `theory/` 某些历史文档描述冲突，以 README 当前版本和受支持脚本为准。
