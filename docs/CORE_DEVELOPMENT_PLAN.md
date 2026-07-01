# 核心开发计划

## 目标顺序

### G1：当前赛题达到 95 分

当前最佳官网结果 V9 为 `88/100`、`327,052` Token、官网 `86.2732`。V10
官网 `85.2928` 对应 `87/100`，已证明旧/新二态分差假设无效。V11 与 V9 同为
`88/100`。V12 把原文语义判断固定为隐藏标签并批量覆盖 5 题，官网降至
`83.3320`、`85/100`，证明该条件化方法无效。修复 MILP 伪可行解并只使用哈希
匹配的 V4-V12 快照后，V13 两项变化的正确数区间为 `88-90/100`。相近 Token 下
综合分达到 90 至少需要 `92/100`，达到 95 仍至少需要 `97/100`。

G1 只推进：

1. PDF 表格、跨页表头和版面结构的确定性恢复；
2. 选项级实体、谓词、数值、日期、条件与例外召回；
3. 显式支持、反证、缺失和冲突的证据充分性判断；
4. 财务、保险和合同的受限计算；
5. 题干范围、否定、全称和复合断言门禁；
6. 官网基线上的分层融合、消融和回归测试。
7. 使用多版本官网正确题数的整数约束排除不可能答案，但不把可行解当作真实标签。
8. 在已确认标签条件下计算逐题、领域和历史答案轨迹的正确数上下界。
9. 差分题显式建模为增益、回归、双错；未经官网验证的标签只能用于候选排序。
10. 隐藏标签、原文语义和模型共识分层存储；语义证据不得进入 MILP 固定条件。
11. 官网快照必须登记 SHA-256、Token、正确题数和可信状态；旧重建快照不得进入硬约束。
12. 候选先计算最坏正确数；批量候选的最坏结果低于当前基线时禁止发布。

G1 的发布条件：

- 候选相对当前官网基线有明确答案 diff；
- 每个变化都有原始页码或条款；
- 约束候选使用的所有提交快照通过注册表哈希校验；
- 候选的最小正确数不低于当前基线，或明确标为单变量诊断实验；
- CSV 通过 100 题完整性、格式和 Token 守恒校验；
- 未经官网验证不得写成准确率提升。

### G1 下一轮方法消融

| 子系统 | 候选技术 | 当前决定 | 验收指标 |
|---|---|---|---|
| 长文档层级 | TreeRAG、LongRefiner、BookRAG | 迁移树结构和双向展开，不接 embedding | 证据 `Recall@K`、父子块噪声率 |
| 异构表格 | TableRAG | 表 schema + 受限 SQLite/DSL 子链 | 表格题执行正确率、口径错误率 |
| PDF 解析 | MinerU、Docling、RAG-Anything parser routing | 仅对复杂/扫描/退化页按质量路由 | 单元格完整率、跨页表头一致率 |
| 版面关系 | LAD-RAG | 迁移跨页/标题/表格符号边，不接向量图 | 多页证据 perfect recall |
| 发布评测 | Active Testing + 版本空间 | 单变量/正交探针，最坏分优先 | 可归因分差、候选下界 |

全量替换 BM25F、全量知识图和全量 VLM 均不进入下一次正式提交。任何新模块先在
固定表格题、跨页题和多跳题子集上做召回与答案双门禁，再决定是否进入主链。

### G2：达到 95 后建设通用 Agent

G2 不再把比赛约束写死在核心层：

- `LLMProvider`：Qwen、OpenAI-compatible、本地推理后端可替换；
- `DocumentDiscovery`：题目无 `doc_ids` 时先做领域和文档级召回；
- `Retriever`：BM25F、dense、late interaction、graph 和 hybrid 通过统一协议组合；
- `Parser`：PyMuPDF/pdfplumber、MinerU/Docling、OCR/VLM 按页面质量路由；
- `DomainAdapter`：财报、合同、法规、保险、研报规则以插件注册；
- `EvaluationSuite`：跨模型、跨档案、跨题型的固定回归集和证据指标。

候选技术栈：

- Qwen3-Embedding/Reranker：多语言 dense retrieval 与二阶段精排；
- ColBERTv2/PLAID：细粒度 late interaction；
- RAPTOR：多层摘要树，用于全局和多跳问题；
- GraphRAG：实体关系和全局主题问题；
- BM25F + dense + reranker + RRF：默认混合检索基线。

这些方向当前只保留接口兼容和调研，不在 G1 达标前替换正式链路。

## 阶段验收

| 阶段 | 硬指标 | 状态 |
|---|---|---|
| G1-A | 官网综合分 `>=90`，约需 `>=92/100` | 进行中 |
| G1-B | 官网综合分 `>=95`，约需 `>=97/100` | 待 G1-A |
| G2-A | 无 `doc_ids` 文档召回 `Recall@10 >= 0.95` | 待 G1 |
| G2-B | 三种模型后端通过同一 100 题回归接口 | 待 G1 |
| G2-C | 新档案只改配置/适配器即可运行 | 待 G1 |
| G2-D | 有 embedding 与无 embedding 模式均可复现 | 待 G1 |

## 参考

- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- [ColBERTv2](https://arxiv.org/abs/2112.01488)
- [RAPTOR](https://arxiv.org/abs/2401.18059)
- [Microsoft GraphRAG Architecture](https://microsoft.github.io/graphrag/index/architecture/)
- [LongRefiner](https://aclanthology.org/2025.acl-long.176/)
- [TreeRAG](https://aclanthology.org/2025.findings-acl.20/)
- [TableRAG](https://aclanthology.org/2025.emnlp-main.710/)
- [RAG-Anything](https://github.com/HKUDS/RAG-Anything)
- [Docling](https://github.com/docling-project/docling)
- [MinerU](https://github.com/opendatalab/MinerU)
- [Active Testing](https://proceedings.mlr.press/v139/kossen21a.html)
