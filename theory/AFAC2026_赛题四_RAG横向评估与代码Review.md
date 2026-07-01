# AFAC2026 赛题四 · RAG 横向评估与代码 Review

> 日期：2026-06-09  
> 评估数据：A 组 100 题；预处理得到 68 份题目关联文档、8110 个 chunk  
> 评估目标：先不考虑 Token，用 A 组题目给定的 `doc_ids` 作为检索命中代理指标，比较不同 RAG 技术栈对“把正确文档送进上下文”的能力。

---

## 1. 代码框架 Review

### 已完成

- 工程骨架已形成：`agent/`、`scripts/`、`tests/`、`processed_data/`、`outputs/`。
- 已实现 V1 主链路：题目读取 → doc_id 映射 → 文档解析 → 分块 → BM25 检索 → 规则压缩 → Qwen 作答 → `answer.csv/evidence.json/token_usage.json`。
- Qwen 客户端已匹配百炼 OpenAI-compatible API：
  - 默认 `qwen3.7-plus`
  - 可切换 `qwen3.7-max`
  - 支持 `extra_body={"enable_thinking": True}`
  - 支持 streaming 和 `reasoning_content` 收集
  - 支持 `stream_options={"include_usage": True}`
- 本地 Key 支持三种方式：环境变量、`.env`、`agent/local_settings.py`。其中 `.env` 和 `agent/local_settings.py` 均被 `.gitignore` 排除。
- 未引入 embedding、向量库、非 Qwen reranker 或非 Qwen 压缩模型。
- 新增横向评估脚本：`scripts/05_compare_rag.py`。
- 新增四个 RAG 框架 lite 变体：`logic_lite_rrf`、`linear_entity_rrf`、`graph_lite_rrf`、`crag_lite`。
- 运行入口 `03/06/07` 已支持逐题 checkpoint 和 `--resume`，适合全量真实 API 调用。
- 多选题默认逐选项判断；证据预算采用领域自适应策略，20 题回归相对 full evidence 无答案变化。

### 测试结果

```powershell
python -m unittest discover -s tests
python -m compileall agent scripts tests
python scripts\01_prepare_docs.py
python scripts\02_build_index.py --tokenizer-mode mixed
python scripts\05_compare_rag.py --tokenizer-modes mixed --output-name rag_compare_mixed_full
python scripts\05_compare_rag.py --tokenizer-modes mixed char word --variants question_options rule_multi_rrf field_boosted_rrf --output-name rag_compare_tokenizers_core
python scripts\05_compare_rag.py --tokenizer-modes mixed --variants question_options rule_multi_rrf field_boosted_rrf logic_lite_rrf linear_entity_rrf graph_lite_rrf crag_lite --output-name rag_compare_frameworks_mixed
```

结果：

- 单元测试：27 个通过。
- 全量预处理：68 documents / 8110 chunks。
- Mixed tokenizer 全策略评估：100 题完成。
- Mixed/char/word 核心策略评估：100 题完成。
- Graph/Logic/Linear/CRAG lite 框架评估：100 题完成。
- 20 题自适应证据预算真实回归：相对 full evidence 答案变化 0，Token 从 158216 降至 145895。
- A 组 100 题真实运行完成：`outputs/a_full_adaptive/answer.csv` 已生成，total_tokens=1002292，未检测到格式/证据/verdict 问题，低置信题为 0。

### Review 发现的问题与风险

| 问题 | 影响 | 处理建议 |
|------|------|----------|
| 评估指标不是最终答案准确率 | 只能衡量“是否检到正确文档”，不能衡量选项判断对错 | 后续需要人工标注小样本或线上 A 榜反馈做 answer-level 评估 |
| 财报 PDF 仍主要靠 PyMuPDF 抽文本 | 表格和扫描页可能丢失关键数值 | V2 优先接 MinerU/PaddleOCR 或 pdfplumber 表格补强 |
| `field_boosted_rrf` 的 boost 规则偏粗 | mixed/char 下反而降低 recall@10 | 保留为实验项，先不要作为默认答题路径 |
| A 组使用 `doc_ids` 限定检索时是 oracle 上界 | B 榜没有 doc_ids，不能依赖这个路径 | V3 必须实现文档级候选检索 |
| 多选题逐选项判断 Token 仍高 | 多选题平均成本显著高于 mcq/tf | 保留逐选项判断，继续优化证据预算和低置信升级，不退回整体判断 |

---

## 2. 横向 RAG 结果

### 2.1 Mixed tokenizer：不同 RAG 策略

| Rank | Variant | hit@1 | hit@5 | hit@10 | recall@10 | all_gold@10 | mrr@10 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | oracle_doc_restricted | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | A 榜上界，不代表 B 榜 |
| 2 | question_options | 0.780 | 0.970 | 1.000 | 0.915 | 0.810 | 0.858 | 历史 sparse baseline，已不再作为正式默认 |
| 3 | rule_multi_rrf | 0.690 | 0.920 | 0.980 | 0.908 | 0.820 | 0.788 | 召回接近，但 top-rank 弱 |
| 4 | field_boosted_rrf | 0.690 | 0.940 | 0.980 | 0.883 | 0.770 | 0.785 | 当前 boost 规则不够稳 |
| 5 | option_rrf | 0.670 | 0.920 | 0.980 | 0.879 | 0.760 | 0.773 | 可作为多选补充检索 |
| 6 | question_only | 0.570 | 0.860 | 0.950 | 0.822 | 0.680 | 0.696 | 明显弱于带选项检索 |

**结论**：当前若只选一个**正式默认检索策略**，应使用 `doc_first_bm25f_expansion`。`question_options` 仍可保留为历史 sparse baseline，但不再代表正式线上默认链路。默认主线现在应以 **BM25F-lite + doc-first local expansion/aggregation + offline expansion-field-enhanced index** 为准。

### 2.2 Tokenizer 技术栈比较

核心策略包括 `question_options`、`rule_multi_rrf`、`field_boosted_rrf`，横向比较 mixed / char / word。

| Rank | Tokenizer | Variant | hit@1 | hit@5 | hit@10 | recall@10 | all_gold@10 | mrr@10 | 结论 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | word | field_boosted_rrf | 0.640 | 0.940 | 1.000 | 0.924 | 0.830 | 0.765 | recall@10 最强，但 top-rank 不强 |
| 2 | mixed | question_options | 0.780 | 0.970 | 1.000 | 0.915 | 0.810 | 0.858 | top-rank 与综合排序最好 |
| 3 | char | question_options | 0.780 | 0.970 | 1.000 | 0.915 | 0.810 | 0.857 | 与 mixed 基本持平 |
| 4 | mixed | rule_multi_rrf | 0.690 | 0.920 | 0.980 | 0.908 | 0.820 | 0.788 | 作为补充召回可用 |
| 5 | word | rule_multi_rrf | 0.710 | 0.950 | 1.000 | 0.902 | 0.780 | 0.805 | word 对部分长合同/财报有帮助 |

**结论**：

- 正式默认路径优先：`doc_first_bm25f_expansion`。
- 追求最大 recall@10 时可并行补充：`word + field_boosted_rrf`。
- `question_options` / `char + question_options` 仍有历史参考价值，但现在应视为 legacy sparse baseline，而不是默认排序。
- `word` 对 recall 有优势，但 hit@1 和 MRR 稍弱，适合作为 RRF 辅助，不建议单独作为默认排序。

### 2.3 按领域观察

| 领域 | 最值得采用的策略 | 观察 |
|------|------------------|------|
| insurance | mixed/char + question_options | hit@1 可达 1.0，但 all_gold@10 约 0.8；多文档保险题仍需多证据合并 |
| regulatory | mixed/char + question_options | hit@1 约 0.95，法规标题和题干匹配较强 |
| research | mixed + question_options / rule_multi_rrf | hit@1 约 0.90-0.95，研报题干实体较明显 |
| financial_contracts | char/word + question_options | hit@10 高，但多份募集书的 all_gold 仍需关注 |
| financial_reports | mixed + rule_multi_rrf | recall@10 可到 0.95，但 hit@1 偏低；财报需要表格/指标索引补强 |

---

## 2.4 GraphRAG / LogicRAG / LinearRAG / CRAG 横向结果

本轮比较的是赛题约束下的 lite 版本，不是原论文完整实现。完整框架路线图见 `theory/AFAC2026_赛题四_RAG框架扩展路线图.md`。

| Rank | Variant | hit@1 | hit@5 | hit@10 | recall@10 | all_gold@10 | mrr@10 | 结论 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | question_options | 0.780 | 0.970 | 1.000 | 0.915 | 0.810 | 0.858 | 历史 sparse baseline，默认口径已切换 |
| 2 | rule_multi_rrf | 0.690 | 0.920 | 0.980 | 0.908 | 0.820 | 0.788 | 补召回可用 |
| 3 | crag_lite | 0.800 | 0.950 | 0.990 | 0.905 | 0.810 | 0.867 | Hit@1/MRR 最好，适合低置信纠错 |
| 4 | logic_lite_rrf | 0.650 | 0.930 | 0.980 | 0.887 | 0.780 | 0.758 | 财报 recall@10 达 0.950，适合逐选项/财报补召回 |
| 5 | field_boosted_rrf | 0.690 | 0.940 | 0.980 | 0.883 | 0.770 | 0.785 | 规则 boost 仍需调参 |
| 6 | graph_lite_rrf | 0.670 | 0.920 | 1.000 | 0.838 | 0.660 | 0.777 | 只能作辅助，不宜默认 |
| 7 | linear_entity_rrf | 0.610 | 0.870 | 0.960 | 0.823 | 0.670 | 0.722 | 需要文档级实体索引配合 |

框架层面的收敛判断：

- `question_options` 不再作为正式默认主路，转为保留的历史 sparse baseline。
- `crag_lite` 保留为低置信纠错路径；它提高 hit@1 和 MRR，但不能单独替代主路。
- `logic_lite_rrf` 对财报更有价值，应接入逐选项判断和 Calculator 前置检索。
- `graph_lite_rrf`、`linear_entity_rrf` 当前规则太轻，不适合全题默认；等 V3 文档级索引/实体图成熟后再升级。

---

## 3. 先不考虑 Token 的推荐默认路径

### V1 默认

1. 正式默认检索主路：`doc_first_bm25f_expansion`
2. legacy sparse 对照：`question_options`
3. 召回补充 / 实验对照：`mixed + rule_multi_rrf`
4. 对合同/财报补充：`word + field_boosted_rrf`
5. 融合方式：RRF 合并多路结果，优先保留来自多个策略共同命中的 chunk/doc
6. 压缩方式：规则抽取式压缩，确保正确文档中的数值、日期、条款号、表格行不会被丢弃

### V2 优先改进

1. 多选题逐选项检索和判断，避免整体 Prompt 漏选。
2. 财报接表格索引和单位归一化，优先提升 hit@1。
3. 合同/保险题加入“必须覆盖所有 gold/candidate docs”的证据配额策略。
4. 低置信题用 `qwen3.7-max` 做二次判断，普通题继续 `qwen3.7-plus`。

---

## 4. 对最终准确率的预期影响

| 改动 | 对 doc 命中的影响 | 对最终答题准确率的预期 |
|------|------------------|------------------------|
| question_only → question_options | recall@10 从 0.822 提升到 0.915 | 显著提升，尤其是选项包含实体/指标时 |
| question_options + rule_multi_rrf 补充 | all_gold@10 可更稳 | 有助于多文档比较和多选题 |
| 单一 mixed → mixed + word 辅助 | recall@10 可从 0.915 提升到 0.924 | 有助于财报/合同，但需防排序噪音 |
| oracle_doc_restricted | 上界 1.0 | A 榜可用，B 榜不可依赖 |
| 表格/KVP 索引 | 目前未完全实现 | 对财报、合同计算题预计收益最大 |

最终建议：**当前正式默认检索应保持 `doc_first_bm25f_expansion`，并把 `question_options` 降级为可保留的历史 sparse baseline / 对照项。** 在不考虑 Token 的前提下，正式默认主线应优先保证长金融文档中的正确文档与正确证据块能稳定上浮，而不是继续沿用仅以 `question_options` 为中心的旧默认叙述。

---

## 5. V10 官网反馈后的评估口径修正

V10 的 `87/100` 证明答案差分中存在“旧、新答案都错”的状态。此后 RAG 横向评估
分为三个层次：

1. 文档召回：`Recall@K`、`all_gold@K`，只衡量是否找到目标文档；
2. 原文证据：逐选项支持、反证、范围条件和数值端点覆盖；
3. 隐藏标签：历次完整提交与官网正确题数构成的三态 MILP 约束。

前两层用于评价 RAG 和推理质量，第三层用于比赛发布门禁。模型共识、GraphRAG、
LogicRAG 或更长上下文均不能替代第三层；排行榜可行解也不能反向证明原文语义。
当前 V11 因此没有更换检索主框架，而是修复评估层对隐藏标签不确定性的错误建模。
