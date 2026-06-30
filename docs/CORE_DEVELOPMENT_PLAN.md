# 核心开发计划

## 目标顺序

### G1：当前赛题达到 95 分

当前最佳官网结果 V9 为 `88/100`、`327,052` Token、官网 `86.2732`。V10
根据 V9 的四增二减模式构造推断 `90/100` 基线，理论综合分 `88.233919`；
该结果必须经官网验证。相近 Token 下综合分达到 90 至少需要 `92/100`，
达到 95 仍至少需要 `97/100`。

G1 只推进：

1. PDF 表格、跨页表头和版面结构的确定性恢复；
2. 选项级实体、谓词、数值、日期、条件与例外召回；
3. 显式支持、反证、缺失和冲突的证据充分性判断；
4. 财务、保险和合同的受限计算；
5. 题干范围、否定、全称和复合断言门禁；
6. 官网基线上的分层融合、消融和回归测试。
7. 使用多版本官网正确题数的整数约束排除不可能答案，但不把可行解当作真实标签。
8. 在已确认标签条件下计算逐题、领域和历史答案轨迹的正确数上下界。

G1 的发布条件：

- 候选相对当前官网基线有明确答案 diff；
- 每个变化都有原始页码或条款；
- CSV 通过 100 题完整性、格式和 Token 守恒校验；
- 未经官网验证不得写成准确率提升。

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
