# AFAC2026 候选方向适配分析（sparse-first / A榜约束版）

> 日期：2026-06-19  
> 目标：不是再列论文库存，而是回答“这些方向在当前 AFAC 主线下到底能不能接、应该接到哪一层、会不会把主线带偏”。  
> 评估口径：
> - 默认主路必须保持 **sparse-first / no-embedding / Qwen-only / 可审计证据**。
> - A 榜主路默认带 **doc_id 约束**，优先看 `doc_ids` 约束下的检索收益，而不是开放域表现。
> - 优先考虑 **当前代码结构可插入点**，避免需要大重构的方案。
> - Token 成本只允许温和增长；不能为了补一点召回把链路变成多轮 agent。

---

## 1. 先给最终判断

### 1.1 最适合当前主线的不是“整框架迁移”，而是 5 类局部增强

1. **文档/粒度路由**：在 A 榜里不改变 doc_id 约束，只决定“在已知文档里先走哪种检索粒度”。
2. **字段感知 chunk 级精排**：把当前 `field_boosted_rrf` 的轻量加分升级为更系统的 sparse scoring。
3. **表格专属检索单元**：把现有 page/table chunk 升级成 row/block 级单元，而不是全文表格块。
4. **受控 retry / evidence-gap 补检索**：保留当前一次 retry 思路，但把“为什么补、补什么”显式化。
5. **索引底座层增强**：BM25F/BM25T 风格索引增强属于高兼容、高收益方向，比继续堆更多 query rewrite 更稳。

### 1.2 五档优先级

| 优先级 | 方向 | 判断 |
|---|---|---|
| P0 | 字段感知 sparse scoring / BM25F-lite | 最符合 sparse-first，几乎不涨 Token，且和现有索引元数据直接兼容 |
| P0 | 表格 row/block 检索单元（T-RAG-lite） | 对财报题最对症，且主要是离线索引改造，不依赖额外模型 |
| P0 | A 榜 doc-scoped granularity routing（SmartChunk-lite） | 不是做开放域文档召回，而是在已知 doc_ids 内决定 section/chunk/table 粒度 |
| P1 | LogicRAG retrieval-first 的 query structure 强化 | 适合复杂题，但必须继续保持实验开关，不应替代默认主路 |
| P1 | evidence-gap-driven 单次 retry（FAIR/CRAG-lite 升级） | 与现有 `crag_lite` / `multi_logicrag_retry` 高兼容，但必须严格控轮数 |
| P2 | 自适应 chunk policy 全面上线 | 研究价值高，但当前代码插点多、离线评估成本高，不应先于上面几项 |
| P2 | 更重的多轮 agent / graph-heavy / iterative search | 与 Token 成本和主线稳定性冲突，不适合作为近期主路 |

一句话结论：
- **近期最值得做的是“索引与检索单元层增强”**，不是继续把更多复杂性放到 query-time agent。  
- **A 榜场景下，很多所谓 doc-level recall 其实应该重写为 doc-scoped routing / coverage / pack selection 问题。**  
- **真正高兼容的升级方向，是少加 Token、多加结构。**

---

## 2. 当前代码约束：哪些是硬边界，哪些是现成插点

## 2.1 已经写死/基本成型的边界

1. `Retriever._candidate_doc_filter()` 在 A 榜 `restrict_to_doc_ids=True` 时直接把检索限制到 `question.doc_ids`，因此 A 榜默认不是开放域 doc recall 问题，而是 **给定文档集合内的 chunk 级召回问题**。见 `agent/retrieve/retriever.py:81-91`。  
2. `BM25SearchIndex.search()` 本质是纯词法 BM25，支持 `filter_doc_ids`，没有 embedding/reranker 依赖。见 `agent/index/bm25.py:123-142`。  
3. `logicrag_qwen_rrf` 已经是 retrieval-first 结构：先 seed query，再拼 query variants、short hypothetical query、sparse feedback query，不直接改 solver 主链。见 `agent/reasoning/logicrag.py:33-65`。  
4. A 榜质量模式已经有逐选项检索和一次 retry 的雏形，特别是 `_solve_multi_logicrag()` 里“不确定选项 -> 扩检索 -> 重判定”的结构。见 `agent/reasoning/solver.py:164-291`。  
5. 现有 chunk schema 已保留 `page / section / clause_id / numbers / dates / chunk_type / caption` 等结构，说明项目已经具备做 field-aware sparse retrieval 的数据基础。见 `agent/schemas.py:97-118`、`agent/index/bm25.py:144-165`。

## 2.2 当前最重要的结构短板

1. `field_boosted_rrf` 只是 RRF 后的 heuristic 加分，还不是字段级独立 scoring；目前只看 clause_id / 数字 / 日期 / title / 选项前缀命中。见 `agent/retrieve/variants.py:159-185`。  
2. 表格 chunk 仍是“caption + table_text”的整块拼接，没有显式的 `table_header / row_label / unit / year_column` 字段，也没有 row/block 级检索单元。见 `agent/preprocess/chunkers.py:106-137`。  
3. 文档级索引主要为 B 榜盲搜准备；A 榜里 doc_ids 已知，因此它在 A 榜里不是主增益位。见 `agent/index/document_index.py:1-4, 62-68`。  
4. `build_retrieval_target()` 已能抽 `must_terms / should_terms / numbers / dates / entities / option_terms`，但这些字段目前主要体现在 query variants 上，还没进入真正的字段化索引打分。见 `agent/retrieve/targets.py:58-107`。  
5. `build_evidence_packs()` 解决的是“命中后如何扩相邻证据”，不是“如何更好命中表格行/条款字段”；因此 pack expansion 不能替代更细的检索单元设计。见 `agent/retrieve/context.py:9-63, 142-185`。

---

## 3. 分类判断一：doc-level recall ideas

先明确：在 A 榜主路里，所谓 doc-level recall 不能按开放域理解。因为 `doc_ids` 已经给定，真正要解决的是：
- 多个 gold docs 之间如何覆盖得更稳；
- 已知文档中先搜哪种粒度；
- 证据 pack 如何保证每个 gold doc 都有代表性命中。

### 3.1 高兼容方向

#### A. doc-scoped coverage routing
- 定义：在已知 `doc_ids` 里，先确保每个目标文档至少有 seed/anchor 命中，再做 chunk 排序。
- 兼容性：很高。
- 原因：当前代码已经有 doc coverage 评估与 evidence pack 的 doc quota 机制，说明主线天然接受“先保覆盖，再做精排”的思路。见 `agent/retrieve/context.py:82-115`、`agent/reasoning/solver.py:188, 229, 264-289`。
- 对 sparse-first 的适配：完全兼容，不需要 embedding。
- Token 成本：接近 0，可纯规则完成。
- 判断：**P0，可直接作为 A 榜增强主线的一部分**。

#### B. SmartChunk-lite 改写成“doc 内粒度路由”而不是“全库检索规划”
- 定义：planner 不负责召回新文档，只负责判断该题优先走 `paragraph / section / table / clause / document summary` 哪类索引或 query shape。
- 兼容性：高。
- 原因：当前 A 榜不是缺文档发现，而是缺“同一文档内该从哪一层证据开始”。`logicrag_qwen_rrf` 与 `build_retrieval_target()` 已具备 query planning 插点。见 `agent/reasoning/logicrag.py:33-65`、`agent/retrieve/targets.py:157-182`。
- 风险：如果做成 Qwen-heavy planner，容易把收益吃掉。
- Token 成本：可控；建议只输出离散标签，不输出长文本计划。
- 判断：**P0，但必须做成低 token 标签路由器**。

### 3.2 中兼容方向

#### C. 更强文档级索引 / doc-level BM25 强化
- 定义：增强 `DocumentSearchIndex`，用更多结构字段改善文档级候选排序。
- 兼容性：中等。
- 原因：这对 B 榜价值更大；A 榜只有在把 `doc_ids` 从 hard constraint 放宽为 hint-only 时才更关键。
- 代码现状：`DocumentSearchIndex` 已存在，但定位是 B 榜粗筛。见 `agent/index/document_index.py:16-68`。
- 判断：**P1 for B 榜，P2 for A 榜主路**。

### 3.3 低兼容方向

#### D. 开放域式 doc recall / 多轮文档扩搜
- 问题：与 A 榜 `doc_ids` 约束天然冲突，且会把问题从“chunk 召回”误改成“外层文档发现”。
- 判断：**不应作为 A 榜 sparse-first 主线方向**。

小结：
- A 榜里的 doc-level recall 最应该落在 **coverage + 粒度路由**，而不是再造一个更复杂的文档发现器。

---

## 4. 分类判断二：chunk-level precision ideas

这是当前 AFAC 最值得投入的层。

### 4.1 最适合当前主线的方向

#### A. BM25F-lite / field-aware sparse scoring
- 定义：把标题、条款号、正文、表格、数字、日期等字段分开建模或分开计分，而不是只在 RRF 后加 heuristic bonus。
- 现有基础：
  - chunk 已有 `clause_id / title / numbers / dates / chunk_type / caption`。见 `agent/index/bm25.py:153-164`。
  - chunk 预处理已经生成 `extra_index_fields`。见 `agent/preprocess/chunkers.py:43, 56, 69`。
  - 技术矩阵已经明确把 BM25F 视为推荐方向。见 `theory/AFAC2026_赛题四_全技术选项矩阵.md:422-426`。
- 兼容性：极高。
- Token 成本：0。
- 工程方式：
  1. 多字段独立索引；或
  2. 单索引但把字段内容复制进带前缀的伪 token；或
  3. 先分别检索 title/clause/table/text，再做 sparse-side fusion。
- 判断：**P0，优先级高于继续增加更多 Qwen 查询改写**。

#### B. Query-aware but lexical-preserving precision routing
- 定义：问题驱动地决定“这一题更该压 clause_id / numbers / dates / option entities 哪类字段”。
- 现有基础：`build_retrieval_target()` 已抽出 must/should/numbers/dates/entities。见 `agent/retrieve/targets.py:72-107`。
- 兼容性：高。
- 风险：若做成复杂 planner，会和上面的 BM25F-lite 职能重叠。
- 判断：**P1，适合作为 BM25F-lite 的 query-side配套，而不是独立大项目**。

### 4.2 不该优先的方向

#### C. 继续堆更多 RRF query 数量
- 问题：当前 `logicrag_qwen_rrf`、`graph_lite_rrf`、`rule_multi_rrf` 已经证明 query 侧扩写并不是无限堆越多越好；过多 query 会增加噪音与检索成本。
- 判断：**不是近期最优解**。应该先增强索引/字段表达，再决定是否扩 query。

小结：
- **chunk-level precision 的最佳路径是“更强的 sparse scoring + 更干净的字段结构”，而不是更重的 agent。**

---

## 5. 分类判断三：table-specific ideas

这是和财报题最强相关的一类，也是当前实现短板最明显的一类。

### 5.1 当前现状

1. 预处理已经单独生成 `chunk_type=table` 的表格 chunk。见 `agent/preprocess/chunkers.py:46-57, 106-137`。  
2. 但单个表格 chunk 还是把 caption 和整体 table_text 拼到一起，缺少 row/column 语义字段。  
3. 技术矩阵已经把 BM25T 明确列为可选的财报专属优化。见 `theory/AFAC2026_赛题四_全技术选项矩阵.md:428-432`。

### 5.2 最适配方向

#### A. T-RAG-lite：row/block 检索单元
- 定义：离线把表格拆成 `表头 + 行标签 + 关键列值 + 单位 + 年份` 的 row/block 级 chunk，而不是整表一块。
- 兼容性：极高。
- 原因：这是纯索引侧改造，不依赖额外模型，也最贴合财报题“年份/指标/数值”三元耦合。
- Token 成本：0。
- 代码结构适配：`build_table_chunk()` 是天然插点；可以在这里新增 table-row chunks 或独立 table index。见 `agent/preprocess/chunkers.py:106-137`。
- 判断：**P0，财报专项最该优先做的方向。**

#### B. page-aware / same-page table expansion
- 定义：命中 table anchor 后，优先扩展同页、相邻正文、caption 相关 chunk。
- 现有基础：`build_evidence_packs()` 对 table/figure 已做 same-page expansion。见 `agent/retrieve/context.py:169-171`。
- 判断：**已有雏形，但这只能补上下文，不足以替代 row-level retrieval。**

### 5.3 中低优先级方向

#### C. graph-aware table prompting
- 定义：把表间关系、跨表映射做成更复杂提示结构。
- 问题：更偏 solver/推理层，Token 增长比索引改造更明显。
- 判断：**P2，只有在 row/block 检索单元已经稳定后才值得做。**

小结：
- 财报题的核心不是“检索到表格”而是“检索到正确的表格局部”。  
- 因此 table-specific 最优动作是 **重做检索单元**，不是重做提示词。

---

## 6. 分类判断四：retry / gap-analysis ideas

这类方向和当前主线兼容，但一定要克制。

### 6.1 当前现状

1. `crag_lite` 的触发逻辑仍偏简单，主要看首二名分差、结果数量和分数。见 `agent/retrieve/variants.py:127-147, 188-196`。  
2. A 榜质量模式下，多选题已有“不确定选项 -> build_retry_queries -> 一次扩检索”的路径。见 `agent/reasoning/solver.py:201-237`。  
3. 这意味着项目不是没有 retry，而是 **已经有 retry 壳子，但缺显式 evidence-gap taxonomy**。

### 6.2 最适配方向

#### A. FAIR/CRAG-lite：evidence-gap typed retry
- 定义：先判断缺口类型，再生成对应补检索，而不是无差别放大 top-k。
- 建议缺口类型：
  - 缺年份
  - 缺主体
  - 缺数值
  - 缺条款例外
  - 缺表头语义
  - 缺对比对象
- 兼容性：高。
- 代码插点：
  - `crag_lite` 的 `_retrieval_is_confident()` 后面；
  - `multi_logicrag.should_expand_uncertain_option()` 前后；
  - `build_retry_queries()` 内部模板化。
- Token 成本：可控，只要坚持“一次补检索”。
- 判断：**P1，高价值，但必须继续限制为 1 次 retry。**

#### B. option-level gap analysis
- 定义：对每个选项分别判“支持充分 / 反证充分 / 证据不足”，不足时再扩检索。
- 兼容性：高。
- 原因：当前多选题主线已经是 option-wise 结构，继续细化 evidence-gap 非常自然。见 `agent/reasoning/solver.py:184-258`。
- 判断：**P1，适合多选题与判断题。**

### 6.3 不建议近期推进的方向

#### C. 多轮 IRCoT / open-ended iterative search
- 问题：Token 成本和执行时延都不稳，且容易越过“acceptable token-cost growth”边界。
- 判断：**不应进入默认主线；最多保留小样本实验。**

小结：
- retry/gap-analysis 是值得做的，但它的正确形态是 **分类更精细、轮数更少**，不是 **轮数更多、agent 更重**。

---

## 7. 分类判断五：index-time vs query-time changes

这是本任务里最关键的一组取舍。

## 7.1 index-time changes：总体更适合 AFAC 当前阶段

### 最值得做
1. **BM25F-lite 字段索引/字段打分**  
2. **table row/block 索引**  
3. **领域内专属字段（标题、条款号、年份、单位、数值、caption）结构化入索引**  
4. **doc-scoped coverage / doc-balanced pack selection 的离线支持字段**

### 原因
- 完全兼容 sparse-first；
- 0 Token；
- 对 A 榜和 B 榜都能复用；
- 不会把系统复杂性全部压到 Qwen 上。

## 7.2 query-time changes：只适合做轻量控制层

### 适合做
1. 粒度标签路由（SmartChunk-lite）  
2. retrieval target 结构化查询（当前已部分具备）  
3. 受控的一次 retry query rewrite  
4. option-level uncertainty expansion

### 不适合做
1. 长链多轮规划  
2. 大量 query 变体堆叠  
3. 依赖长摘要/长记忆的复杂 agent loop

## 7.3 总体判断

如果只问“下一步应该把工程时间花在哪一侧”，我的判断是：

- **70% 时间应该放在 index-time / retrieval-unit 改造；**
- **30% 时间放在 query-time 的轻量路由和低置信补检索。**

原因很简单：
- 当前代码已经证明 query-time 结构化能力不是 0；
- 真正还没补上的，是 sparse 底座对字段、表格和文档局部结构的表达力。

---

## 8. 推荐的近期实验顺序（按 AFAC 适配优先级）

## 8.1 第一梯队：最值得立刻进入实验

### A. BM25F-lite / field-aware sparse scoring
实验问题：相比 `field_boosted_rrf`，真正字段化 scoring 是否能稳定提升 `hit@1 / mrr@10`，同时不拖累 `all_gold@10`？

为什么排第一：
- 零 Token
- 与当前 schema 最贴
- 对 insurance / regulatory / contracts / reports 都有收益面

### B. T-RAG-lite / table row-block index
实验问题：财报题是否因 row/block 单元而显著提升 `recall@10 / all_gold@10`？

为什么排第二：
- 当前实现短板最明确
- 对 reports 域最可能有立竿见影收益

### C. doc-scoped granularity routing
实验问题：在已知 doc_ids 下，优先走 table/clause/paragraph 粒度，是否优于统一 chunk 检索？

为什么排第三：
- 很贴 A 榜实际问题
- 能复用 SmartChunk 思想，但不把问题做重

## 8.2 第二梯队：在第一梯队稳定后再接

### D. evidence-gap typed retry
实验问题：与当前 `crag_lite` / `multi_logicrag_retry` 相比，显式 gap 类型是否能减少无效 retry？

### E. LogicRAG query structure 强化
实验问题：更好的 query DAG / rank merge / pruning 是否真的优于现有 `logicrag_qwen_rrf`，且 Token 成本仍可接受？

## 8.3 第三梯队：暂不优先

### F. 自适应 chunk policy 全量上线
先做离线分析，不应一开始就改全语料 chunking 主路。

### G. 多轮 agent / graph-heavy retrieval
仅保留研究对照价值，不适合近期默认路线。

---

## 9. 最终建议：把每类候选方向翻译成 AFAC 可执行语言

如果要把这次分析翻译成项目里的具体工程语言，我建议这样表述：

1. **doc-level recall ideas**  
   在 A 榜里改写为：`doc coverage + doc-scoped granularity routing`，而不是开放域 doc discovery。

2. **chunk-level precision ideas**  
   重点做：`field_boosted_rrf -> BM25F-lite / field-aware sparse scoring`。

3. **table-specific ideas**  
   重点做：`whole-table chunk -> row/block retrieval units`。

4. **retry/gap-analysis ideas**  
   重点做：`score-gap retry -> typed evidence-gap retry`，且最多 1 次。

5. **index-time vs query-time**  
   当前阶段优先顺序应是：
   `index-time structure upgrade > query-time light routing > multi-round agentization`

我的最终结论：
- **最 AFAC-fit 的路线不是“更聪明地问更多次”，而是“让 sparse 检索底座先更懂字段、表格和文档局部结构”。**
- **如果只能押三条线，就押：BM25F-lite、T-RAG-lite、doc-scoped granularity routing。**
- **如果还允许第四条线，再加 evidence-gap typed retry；LogicRAG 更适合作为这几条底座能力之上的控制层，而不是继续单独膨胀。**
