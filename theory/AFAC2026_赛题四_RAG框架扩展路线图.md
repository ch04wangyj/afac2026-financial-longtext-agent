# AFAC2026 赛题四 · RAG 框架扩展路线图

> 日期：2026-06-09  
> 目标：把 GraphRAG、LogicRAG、LinearRAG、CRAG、LightRAG、HippoRAG、RAPTOR、KAG/OpenSPG、Self-RAG、IRCoT 等方向拆成“原框架能力”和“本赛题可落地版本”，并持续同步到代码 variants 与评估报告。

---

## 1. 当前约束下的结论

本赛题不是通用开放域 QA，而是金融长文档选择题/判断题。约束是：不使用 embedding、不加载非 Qwen 模型、不微调、需要输出可审计证据。因此不能直接照搬大部分现代 RAG 开源框架。

规则核对后的结论：本文件只保留技术调研和实验分支说明，默认工程不得被这些通用框架带偏。官方核心是 **Qwen-only、禁止 embedding、动态压缩、严格 Token 统计、A/B 榜差异处理**。

当前应采用：

1. 默认主路：`mixed + question_options`，因为 A 组 doc_id 代理评估里综合命中最稳。
2. 纠错主路：`crag_lite`，当首轮检索置信不足时触发图式/规则补检索。
3. 多跳主路：`logic_lite_rrf`，先用规则实体和选项构造子查询；后续按 staged rollout 演进到 `logicrag_qwen_rrf`，不直接跳到 full-agent。
4. B 榜盲搜候选：`linear_entity_rrf` + 文档级索引，后续演进成 relation-free hierarchical graph。
5. 图谱路线：先做 `graph_lite_rrf` 的实体共现边，不急着上完整 GraphRAG/OpenSPG。

当前不建议作为默认，且不得直接接入原版实现：

- 原版 LightRAG、HippoRAG、RAPTOR：核心依赖 vector/embedding、聚类或额外模型组件，和赛题限制冲突。
- 原版 Self-RAG：需要训练带 reflection token 的模型，不适合只用 Qwen API 的比赛链路。
- 完整 OpenSPG/KAG：金融领域价值高，但工程重，应作为 V3/V4 专项，不应阻塞 V1/V2；若实现，也只能先做 Python 规则 KVP，不部署非必要图谱系统。

---

## 2. 框架横向矩阵

| 框架 | 原论文/开源核心 | 对本赛题的价值 | 风险 | 当前处理 |
|------|-----------------|----------------|------|----------|
| BM25/RRF Baseline | 稀疏词法检索、多路排序融合 | 合规、可解释、速度快 | 对跨表格/跨文档推理弱 | 默认主线 |
| GraphRAG | LLM 抽实体图、社区摘要、全局问答 | 适合全局/跨文档主题题 | 建图和摘要 Token 成本高；社区摘要可能丢数值 | 只保留实体共现 lite |
| LightRAG | 图结构索引 + 双层检索 + vector 表示 | 增量更新和实体关系召回有价值 | 原版使用 vector representations | 不原样接入，只借鉴双层检索 |
| HippoRAG | LLM + KG + Personalized PageRank | 多跳实体关系召回有价值 | KG/PPR/模型组件较重，可能依赖 embedding | 暂缓，待实体图稳定后接 PPR-lite |
| RAPTOR | 递归聚类 + 抽象摘要树 | 长文档层级摘要有吸引力 | 聚类/embedding/摘要都高风险，数值题容易摘要丢失 | 暂不采用 |
| LogicRAG | 查询时构造推理 DAG，不依赖预构图 | 最契合多选/多跳/逐选项判断 | 需要可靠子问题生成和剪枝 | 已实现 `logic_lite_rrf` |
| LinearRAG | 轻量实体抽取、relation-free hierarchical Tri-Graph | 适合 B 榜大规模盲搜和低成本索引 | 原版有 semantic linking，需替换为词法/规则链接 | 已实现 `linear_entity_rrf` |
| CRAG | 检索质量评估 + 纠错动作 + 选择性过滤 | 检索失败时触发补救，很适合比赛 | 原版可能用外部 web；本赛题只能在给定文档内纠错 | 已实现 `crag_lite` |
| KAG/OpenSPG | 专业领域知识图谱、事实+逻辑融合 | 金融/监管领域上限高 | Java/Scala 图谱系统较重，搭建成本高 | V3/V4 专项 |
| Self-RAG | 训练模型生成 reflection tokens，自适应检索/批判 | 自检思想可借鉴 | 原版需要训练/特定模型 | 只借鉴低置信自检 |
| IRCoT | 检索与 CoT 推理交替进行 | 多步题可逐步补证据 | 迭代调用增加 Token | V2 低置信触发 |

---

## 3. 当前代码已经纳入的 lite 版本

代码位置：

- `agent/retrieve/structured_queries.py`
- `agent/retrieve/variants.py`
- `tests/test_structured_rag_variants.py`

新增 variant：

| Variant | 对应方向 | 实现方式 | 适用题型 |
|---------|----------|----------|----------|
| `logic_lite_rrf` | LogicRAG-lite | 从题干/选项抽法规名、年份、实体、指标、数值，构造子查询并 RRF | 多选、跨条件、需要逐选项验证 |
| `linear_entity_rrf` | LinearRAG-lite | 把高信号实体线性展开为多条查询 | B 榜盲搜、研报/财报实体题 |
| `graph_lite_rrf` | GraphRAG-lite | 用实体 pair 近似共现边，不建 LLM 社区摘要 | 多实体、多文档比较 |
| `crag_lite` | CRAG-lite | 首轮 `question_options`，分差不足则触发 graph/rule 补召回 | 检索低置信题 |

这些都是无 embedding、无非 Qwen 模型、可审计的替代实现。

---

## 4. 最新 A 组代理评估

命令：

```powershell
python scripts\05_compare_rag.py --tokenizer-modes mixed --variants question_options rule_multi_rrf field_boosted_rrf logic_lite_rrf linear_entity_rrf graph_lite_rrf crag_lite --output-name rag_compare_frameworks_mixed
```

结果使用 A 组题目给定的 `doc_ids` 作为检索命中代理指标，不是最终答案准确率。

| Rank | Variant | hit@1 | hit@5 | hit@10 | recall@10 | all_gold@10 | mrr@10 | 当前判断 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `question_options` | 0.780 | 0.970 | 1.000 | 0.915 | 0.810 | 0.858 | 默认主路仍最稳 |
| 2 | `rule_multi_rrf` | 0.690 | 0.920 | 0.980 | 0.908 | 0.820 | 0.788 | 补召回可用，top-rank 弱 |
| 3 | `crag_lite` | 0.800 | 0.950 | 0.990 | 0.905 | 0.810 | 0.867 | Hit@1/MRR 最好，可做纠错主路 |
| 4 | `logic_lite_rrf` | 0.650 | 0.930 | 0.980 | 0.887 | 0.780 | 0.758 | 财报召回有增益，适合逐选项前置 |
| 5 | `field_boosted_rrf` | 0.690 | 0.940 | 0.980 | 0.882 | 0.770 | 0.785 | boost 规则需继续调 |
| 6 | `graph_lite_rrf` | 0.670 | 0.920 | 1.000 | 0.838 | 0.660 | 0.777 | 只能作辅助，不可默认 |
| 7 | `linear_entity_rrf` | 0.610 | 0.870 | 0.960 | 0.823 | 0.670 | 0.722 | 当前规则抽实体太粗，需文档级索引配合 |

领域观察：

- 财报：`logic_lite_rrf` 和 `rule_multi_rrf` 的 recall@10 达 0.950，高于 `question_options` 的 0.825。
- 保险/监管/研报：`question_options` 仍非常强，Hit@1 可达 0.95-1.00。
- 合同：`crag_lite` 和 `question_options` 都较稳，但 all_gold@10 仍需证据配额策略。

---

## 5. 迭代路线

### V1.1：保持主线稳定

- 默认 `question_options`。
- `crag_lite` 只在首轮低置信题触发。
- `logic_lite_rrf` 仅作为多选/财报候选补召回。
- `graph_lite_rrf`、`linear_entity_rrf` 不进入默认排序，只保留实验开关。

### V2：Qwen 驱动的动态结构（按阶段推进，不替换默认主路）

#### V2-Stage 0：文档化与边界固化

- 先在 `theory/references/notes/` 固化 LogicRAG 适配结论：只迁移“运行时 DAG / 分层检索 / pruning / rolling memory”思想。
- 明确红线：**Qwen-only、BM25-only、可审计证据、不得引入 embedding/reranker/外部图系统**。
- 同步废弃旧 `full evidence / adaptive evidence` 方案的 baseline 地位：它们只保留为历史记录，不再参与当前 ARS/LogicRAG 的正式 stop/go 判断。
- 该阶段不接入答题链路，只统一实验口径与 stop/go 阈值。

#### V2-Stage 1：`logicrag_qwen_rrf`（retrieval-first）

- 让 Qwen 为题目生成 3-6 个子问题、依赖边、可裁剪节点。
- 本地做 DAG 校验、去重、拓扑分层与同层 query merge。
- 检索侧仍只用 BM25 / 文档级 BM25 / RRF，不改 solver 主链。
- 评估口径优先看 A 组 `doc_ids` 代理指标：`all_gold@10`、`recall@10`、`mrr@10`、`hit@1`。
- 若相对 `logic_lite_rrf` 没有稳定增益，则停留在 Stage 1，不继续扩大范围。

#### V2-Stage 2：逐选项判定版 LogicRAG

- 只在 retrieval-first 达标后推进。
- 多选/判断题改成“公共上游节点 + 选项特异节点”的 DAG。
- 公共节点负责文档、主体、年份、章节定位；选项节点负责数值、条件、例外条款验证。
- 输出仍交给现有 solver 消化，不做大规模推理框架替换。

#### V2-Stage 3：rolling memory + 低置信回补

- 给每一层增加短摘要记忆：已确认事实、未解决空白、下层检索约束。
- 与 `crag_lite` 协同：CRAG 负责决定“是否回补”，LogicRAG 负责决定“回补什么子问题”。
- 只允许 1 次受控补检索，避免变成高成本多轮代理。

#### V2-Stage 4：small live run

- 仅在小样本上开启，不作为默认 variant。
- 先跑 smoke/sample，对照 `question_options`、`crag_lite`、`logic_lite_rrf`。
- 观察最终答案质量、证据可审计性、平均 token、单题耗时、回退率。
- 只有在收益稳定且成本可控时，才考虑扩大到更大规模实验。

#### V2 同步项

- CRAG-lite 升级：检索质量从 score gap 扩展到“规则分数 + Qwen 低成本判别”。
- IRCoT-lite：只在低置信题启用“子结论 → 补检索 → 再判断”1-2 轮，不进入默认主路。

### V3：B 榜盲搜和图谱

- LinearRAG-lite 升级为文档级实体倒排 + 文档重要性聚合。
- GraphRAG-lite 升级为实体共现图 + BM25 边权 + PPR-lite。
- KAG/OpenSPG-lite：先不部署 OpenSPG，用 Python 规则 KVP 表达“法规-条款-条件-例外”“财报-指标-年份-数值”。
- 只有当 Python 图谱验证显著提升，再考虑完整 OpenSPG。

---

## 6. 文献与开源来源

- GraphRAG: https://arxiv.org/abs/2404.16130
- LogicRAG: https://arxiv.org/abs/2508.06105
- LinearRAG: https://arxiv.org/abs/2510.10114
- LightRAG: https://arxiv.org/abs/2410.05779
- HippoRAG: https://arxiv.org/abs/2405.14831
- RAPTOR: https://arxiv.org/abs/2401.18059
- CRAG: https://arxiv.org/abs/2401.15884
- Self-RAG: https://arxiv.org/abs/2310.11511
- IRCoT: https://arxiv.org/abs/2212.10509
- OpenSPG/KAG: https://github.com/OpenSPG/openspg
