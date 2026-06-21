# AFAC2026 检索增强文献清单（Phase 2）

> 日期：2026-06-19  
> 目标：围绕 AFAC2026 当前检索瓶颈，整理一份“先看什么、可借什么、不该直接搬什么”的论文库存。  
> 口径：优先看是否能在 **Qwen-only / lexical-only / 可审计证据 / 金融长文档选择题** 约束下落地，而不是看开放域 SOTA。

---

## 1. 先给结论

当前最值得优先吸收的不是“整套新框架”，而是 5 类可拆用能力：

1. **文档/查询自适应分块**：减少固定 chunk size 对财报、条款、表格混合文档的误伤。  
2. **按题目动态决定检索粒度**：不同题型不该统一走同一层 chunk。  
3. **表格感知的多阶段检索**：财报/指标题不能把表头、行值、跨表关系打散。  
4. **证据缺口驱动的补检索**：低置信题不该“整题重跑”，而该明确缺什么再补什么。  
5. **字段感知的稀疏检索**：标题、条款号、年份、数字、表头应带不同权重。

对 AFAC2026 的优先级判断：

- **P0（最该尽快吸收）**：`2602.22225v1` SmartChunk Retrieval、`2504.01346v4` T-RAG、`2508.06105v2` LogicRAG
- **P1（作为纠错/回补思想来源）**：`2510.22344v1` FAIR-RAG、`2401.15884v1` CRAG、`2212.10509v1` IRCoT
- **P2（更偏离当前主线，但可提供分块评估框架）**：`2603.25333v1` Adaptive Chunking

一句话判断：
- 如果只能先做一个方向，优先做 **“SmartChunk 的粒度规划思想 + T-RAG 的表格分层检索思想 + LogicRAG 的运行时子问题结构”**。  
- 如果只能先做一个低风险增强，优先做 **字段感知 sparse re-rank / boost + 表格专属 chunk/unit**。  
- 如果要做低置信补救，优先做 **CRAG/FAIR-RAG 风格的 evidence-gap 触发器**，不要直接上多轮 agent。

---

## 2. AFAC2026 当前瓶颈映射

结合仓库现状，当前外部文献应主要服务于以下瓶颈：

| AFAC 瓶颈 | 当前仓库对应现象 | 文献最相关方向 |
|---|---|---|
| 固定 chunk 切法对财报/法规/合同不统一 | 目前主路仍以 BM25 + 既有 chunk 为主，固定切分容易把表头、条件、年份关系切散 | Adaptive Chunking, SmartChunk |
| 同一题型需要不同检索粒度 | `question_options` 稳，但复杂多条件题/财报题召回不总是最优 | SmartChunk, LogicRAG |
| 表格证据容易丢语义 | 财报题往往需要“表头 + 行值 + 年份 + 指标”联动 | T-RAG |
| 首轮检索失败时补救不够结构化 | 当前已有 `crag_lite`，但更像规则 fallback，缺“缺什么证据”级别的显式建模 | FAIR-RAG, CRAG, IRCoT |
| 标题/条款号/数字/日期/表头权重不均 | 当前已有 `field_boosted_rrf`，但本质还是轻量 heuristic boost | BM25F/BM25T 风格 field-aware sparse retrieval |

和仓库现状的直接连接：
- `agent/retrieve/variants.py` 已有 `field_boosted_rrf`、`logic_lite_rrf`、`logicrag_qwen_rrf`、`crag_lite`。这意味着文献调研最有价值的，不是“要不要做这些方向”，而是“怎么把它们做得更像论文里的强版本，但不越过比赛红线”。
- `theory/AFAC2026_赛题四_全技术选项矩阵.md` 已明确把 **BM25F（字段加权）**、**BM25T（表格单元）** 视为和赛题高度相关的 sparse 路线；因此 field-aware sparse retrieval 在本项目里不是空白方向，而是已具备工程落点。

---

## 3. 必看论文逐篇判断

## 3.1 2603.25333v1 — Adaptive Chunking: Optimizing Chunking-Method Selection for RAG
- 时间：2026-03-26
- 论文主张：不是所有文档都该用同一种 chunking；先用一组文档级 intrinsic metrics 评估 chunk 质量，再为不同文档挑更合适的切分方式。
- 和 AFAC 的直接相关点：
  1. AFAC 语料天然异构：监管条文、保险条款、财报正文、财报表格、研报叙述，固定切分很容易“一刀切失真”。
  2. 它强调 **document-aware**，这和 AFAC 的 domain/document type 差异很匹配。
  3. 它给的是“如何比较 chunking”的框架，而不是只能在某个 embedding pipeline 里生效。
- 局限：
  1. 更像“分块选择框架”，不是完整检索方案。
  2. 论文收益建立在整体 RAG 输出提升上，未直接证明对 BM25-only 中文金融长文档同样稳定。
  3. 若完全照搬其指标，需要额外实现/评估开销。
- 对 AFAC 的落地建议：
  - 不直接照搬全部指标体系；先把它降成 **离线 chunk policy 对比工具**。
  - 优先比较 3 类文档：法规/合同、财报正文、财报表格。
  - 先回答“不同文档类型是否真的该用不同 chunk 策略”，再决定是否做自适应路由。
- 优先级：P2（值得吸收评估思路，但不是最先上线的检索增益点）

## 3.2 2602.22225v1 — SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document RAG
- 时间：2025-12-17
- 论文主张：问题不只在文档怎么切，还在 **每个 query 应该取哪一层 chunk 抽象粒度**；通过 planner 动态决定检索粒度，并用轻量压缩模块避免反复总结。
- 和 AFAC 的直接相关点：
  1. 这是当前清单里最贴合 **“query-aware chunk granularity planning”** 的论文。
  2. AFAC 题型差异极大：有些题要精确条款片段，有些题要章节级上下文，有些题要表格行级证据；统一 chunk size 本来就不合理。
  3. 它把“检索粒度选择”显式化，这比简单多路召回更接近我们想做的 `logicrag_qwen_rrf` 上游规划层。
- 局限：
  1. 原论文明显不是为 lexical-only / BM25-only 约束设计的。
  2. 其 planner + compression 模块若完整照搬，工程上会偏重。
  3. reinforcement learning 的 planner 训练路径不适合比赛主线。
- 对 AFAC 的落地建议：
  - 只迁移 **“按题决定粒度层级”** 这件事，不迁移其训练方案。
  - 可以做成 Qwen 输出的轻量规划标签，例如：`span-level / paragraph-level / section-level / table-row-level / document-level`。
  - 首版甚至不需要新模型，只需让 planner 决定：先搜细粒度 index、章节级 index，还是表格行级 index。
- 优先级：P0（当前最像“可直接转成工程实验”的外部论文）

## 3.3 2504.01346v4 — RAG over Tables: Hierarchical Memory Index, Multi-Stage Retrieval, and Benchmarking
- 时间：2025-04-02
- 论文主张：表格 RAG 不是普通文本 RAG；需要 table-corpora-aware 的结构，包括 hierarchical memory index、多阶段检索、图感知提示，并给出 MultiTableQA 基准。
- 和 AFAC 的直接相关点：
  1. 这是当前清单里最直接解决 **“table-aware multi-stage retrieval”** 的论文。
  2. AFAC 财报题经常依赖“表头 + 年份列 + 指标行 + 跨表映射”，普通 chunk 容易把这些关系切断。
  3. 它明确把“先筛表、再筛表内信息、再推理”拆成多阶段，这和 AFAC 的财报题检索链高度一致。
- 局限：
  1. 原论文是 table corpora 场景，不完全等价于“长 PDF 中夹杂表格”的 AFAC 输入形态。
  2. graph-aware prompting 的一部分可能超出我们当前最小可行路线。
  3. 若直接追求完整 T-RAG，工程范围会偏大。
- 对 AFAC 的落地建议：
  - 先借其 **检索单元设计**：把“表头 + 行标题 + 单元格值 + 年份/币种/单位”重组为可检索 table row / block。
  - 再借其 **多阶段思路**：先文档/章节定位，再表格定位，再行/列定位。
  - 对财报 domain 单独试一个 table-aware index，而不是全语料统一改造。
- 优先级：P0（对财报题最可能有立竿见影价值）

## 3.4 2510.22344v1 — FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation
- 时间：2025-10-25
- 论文主张：核心不是盲目迭代，而是先做 Structured Evidence Assessment，把“已确认事实”和“还缺什么证据”显式列出来，再基于缺口生成下一轮子查询。
- 和 AFAC 的直接相关点：
  1. 这是当前清单里最贴合 **“evidence-gap-driven retry / iterative refinement”** 的论文。
  2. AFAC 很多失败题不是“完全没召回”，而是“召回了部分证据，但缺年份/条件/例外/对比对象”。
  3. 这种“缺口驱动补检索”比简单扩大 top-k 或整题重跑更适合成本受控的比赛链路。
- 局限：
  1. 原论文服务多跳 QA，默认允许更强的迭代式 agent 行为。
  2. 若不约束轮数，很容易把成本拉高。
  3. SEA 的完整实现可能过重。
- 对 AFAC 的落地建议：
  - 不做多轮 agent，只保留 **一次受控补检索**。
  - 让 `crag_lite` 或未来的低成本判别器先判断“是否值得补检索”，再让 SEA-lite 输出缺口类型：`缺年份 / 缺主体 / 缺数值 / 缺条款例外 / 缺表头语义`。
  - 对多选题尤其适合：每个选项单独标“证据充足 / 证据缺口”。
- 优先级：P1（很适合做 CRAG-lite 的升级版思想来源）

## 3.5 2508.06105v2 — LogicRAG: Retrieval Augmented Generation with Adaptive Reasoning Structures
- 时间：2025-08-08
- 论文主张：不预构图，而是在 query time 把问题拆成子问题并构造依赖 DAG，再进行 graph pruning / context pruning，以更低 token 成本完成复杂检索推理。
- 和 AFAC 的直接相关点：
  1. 它已经被仓库路线图识别为最契合多选/多跳/逐选项判断的外部框架来源。
  2. AFAC 的很多复杂题本质上不是“多文档开放问答”，而是“先定位范围，再验证选项条件”。这种结构非常适合 DAG 化。
  3. 它天然适合作为 `logic_lite_rrf -> logicrag_qwen_rrf` 的升级理论基础。
- 局限：
  1. 原论文并不是围绕中文金融 BM25-only 场景设计。
  2. 如果一上来就做 full-agent，会偏离当前稳定主路。
  3. 真正难点不在 DAG 形式，而在子问题质量、合并、剪枝、回退策略。
- 对 AFAC 的落地建议：
  - 继续坚持当前仓库已有判断：先做 retrieval-first，不改 solver 主链。
  - 让 Qwen 只生成 3-6 个子问题 + 依赖边 + 可裁剪节点；检索仍用 BM25/RRF。
  - DAG 更适合作为多选题、财报题、跨条件监管题的“高价值实验 variant”，不该强推全题默认。
- 优先级：P0（当前最符合已有路线图，也最容易与现有 variant 对接）

---

## 4. 补充候选：虽然不是任务硬性要求，但建议一并纳入库存

## 4.1 2401.15884v1 — CRAG: Corrective Retrieval Augmented Generation
- 价值：把“先评估首轮检索是否可靠，再决定是否补救”这件事做成独立模块。
- 对 AFAC 的意义：与当前 `crag_lite` 方向高度同构，是现有 fallback 设计最直接的论文锚点。
- 建议借用：retrieval evaluator + selective filtering 思想。
- 不建议借用：依赖外部 web 搜索的部分。

## 4.2 2212.10509v1 — IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning
- 价值：强调“下一轮该检索什么，取决于上一轮已经得到了什么结论”。
- 对 AFAC 的意义：适合作为低置信多步题的受控补检索思想来源。
- 建议借用：1-2 轮受控“子结论 -> 补检索 -> 再判断”。
- 不建议借用：开放式、长链 CoT 驱动的高成本流程。

## 4.3 2412.12881v1 — RAG-Star
- 价值：把检索接入 deliberative reasoning / verification / refinement。
- 对 AFAC 的意义：更像远期研究对照组，不像近期工程主线。
- 建议借用：verification / refinement 视角。
- 不建议借用：MCTS 式重推理框架，成本过高。

---

## 5. 对“field-aware sparse retrieval”的单独判断

这次硬性要求里最容易缺失的，其实不是 chunking 或 iterative refinement，而是 **field-aware sparse retrieval**：

1. 在本次必看 5 篇里，**没有一篇是专门围绕 BM25F / fielded sparse retrieval 写的直接对口论文**。  
2. 但 AFAC 这条线并不空白：仓库已经有 `field_boosted_rrf`，并且 `theory/AFAC2026_赛题四_全技术选项矩阵.md` 已明确把 **BM25F（标题/条款号/正文/表格字段加权）**、**BM25T（表头+行值作为表格检索单元）** 视为高相关方向。  
3. 因此对 AFAC 而言，field-aware sparse retrieval 的正确动作不是“等一篇完美外部论文”，而是：
   - 把当前 heuristic boost 升级为更系统的字段打分；
   - 把 `title / clause_id / dates / numbers / table_header / row_label` 从 metadata hint 提升为真正的可控检索字段；
   - 在财报表格域，把 BM25T 风格“表头+行”单元纳入独立索引。

换句话说：
- **SmartChunk / LogicRAG / FAIR-RAG** 更像“检索过程控制层”的外部增量；
- **BM25F / BM25T 风格设计** 更像“稀疏索引底座层”的内部必补课。

---

## 6. 推荐的落地顺序（只按 AFAC 约束排序）

## 6.1 第一梯队：最值得直接进实验计划

### A. Query-aware granularity planning（来自 SmartChunk）
最适合转成一个低风险实验：
- 输入：题干 + 选项
- 输出：推荐检索粒度标签（细 chunk / 段落 / 章节 / 表格行 / 文档级）
- 检索：仍是 BM25 / 文档 BM25 / RRF
- 价值：直接回答“同一题是否该走不同粒度 index”

### B. Table-aware multi-stage retrieval（来自 T-RAG）
最适合财报专项：
- 先定位文档/章节
- 再定位表格
- 再定位行列/年份/指标
- 价值：正中财报题痛点

### C. Retrieval-first adaptive reasoning structure（来自 LogicRAG）
最适合复杂多选/跨条件题：
- 先用 Qwen 规划子问题
- 再用 BM25/RRF 执行
- 只做 retrieval-first，不改 solver

## 6.2 第二梯队：作为低置信补检索升级

### D. Evidence-gap-driven retry（来自 FAIR-RAG + CRAG + IRCoT）
适合作为 `crag_lite` 升级项：
- 先判断是否低置信
- 再标记缺口类型
- 最后做 1 次受控补检索

## 6.3 第三梯队：作为底层索引改造

### E. Field-aware sparse retrieval（来自内部 BM25F/BM25T 路线）
这条虽然“论文感”没前几项强，但工程收益可能很现实：
- 标题、条款号、日期、数字、表头加权
- 表格单独索引
- 财报 domain 专属 row/block 检索单元

---

## 7. 最终建议：这份库存如何服务下一步任务

如果下一阶段只允许开 3 条实验线，我建议：

1. **SmartChunk-lite**：只做“粒度选择器”，不做 RL planner。  
2. **T-RAG-lite**：只做财报表格专属检索单元 + 多阶段检索。  
3. **FAIR/CRAG-lite**：只做“证据缺口分类 -> 一次补检索”。

如果下一阶段更偏路线图主线衔接，则建议：

1. 继续推进 `logicrag_qwen_rrf` 的 retrieval-first 实验；  
2. 同步补上 `field_boosted_rrf -> field-aware sparse scoring`；  
3. 对财报域单开 table-aware index，而不是等全域统一改造。

我的总体判断：
- **最应该优先读透并转实验的是 SmartChunk、T-RAG、LogicRAG。**
- **最应该转成低成本稳健机制的是 FAIR-RAG / CRAG。**
- **最应该尽快落成工程底座的，是 BM25F/BM25T 风格的 field-aware sparse retrieval。**

---

## 8. 本次库存涉及的核心论文链接

- 2603.25333v1 — Adaptive Chunking: https://arxiv.org/abs/2603.25333v1
- 2602.22225v1 — SmartChunk Retrieval: https://arxiv.org/abs/2602.22225v1
- 2504.01346v4 — RAG over Tables / T-RAG: https://arxiv.org/abs/2504.01346v4
- 2510.22344v1 — FAIR-RAG: https://arxiv.org/abs/2510.22344v1
- 2508.06105v2 — LogicRAG: https://arxiv.org/abs/2508.06105v2
- 2401.15884v1 — CRAG: https://arxiv.org/abs/2401.15884v1
- 2212.10509v1 — IRCoT: https://arxiv.org/abs/2212.10509v1
- 2412.12881v1 — RAG-Star: https://arxiv.org/abs/2412.12881v1
