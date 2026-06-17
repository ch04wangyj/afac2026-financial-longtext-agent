# LogicRAG 论文适配结论（AFAC2026）

> 论文：*You Don’t Need Pre-built Graphs for RAG: Retrieval Augmented Generation with Adaptive Reasoning Structures*  
> 结论口径：只讨论其方法思想如何适配当前 AFAC2026 仓库与赛题约束，不直接照搬原始实现。

---

## 1. 论文核心

LogicRAG 的关键价值不在“图数据库”或“预构图”，而在 **query-time adaptive reasoning structure**：

1. **把复杂问题拆成可检索子问题**：把一个多条件、多跳问题拆成 3-6 个更易召回的子问题。
2. **在运行时动态生成依赖 DAG**：不是离线建图，而是围绕当前题目临时生成“先查什么、再查什么”的依赖结构。
3. **按拓扑层做检索与求解**：先解决上游节点，再让下游节点复用上游结论，降低检索发散。
4. **graph pruning / unified querying**：合并同层或近义子问题，避免重复检索。
5. **context pruning / rolling memory**：每层只保留后续判断真正需要的证据和中间结论，压缩 token。
6. **最终组合答案**：把各节点得到的事实证据汇总到最终问答或判定。

对本项目最有启发的是：**把“复杂题拆成可审计的检索-判断链”，而不是一次性把整题丢给大模型。**

---

## 2. 与 AFAC2026 约束的冲突点

## 2.1 检索机制冲突

原论文实验依赖 embedding 检索与对应评测设定；当前项目明确以 **BM25 / 词法检索 / RRF** 为主，且已有 `question_options`、`logic_lite_rrf`、`crag_lite` 等无 embedding 路线。

**结论**：原论文“动态结构”可用，但其检索底座必须替换成 lexical-only。

## 2.2 模型使用冲突

原论文实验并非围绕 **Qwen-only** 比赛限制设计；本项目必须遵守：

- 只走 Qwen（DashScope）
- 不能引入额外生成模型或 reranker
- 需要可审计、可复现、可控 token 成本

**结论**：规划、压缩、低置信复核都要由 Qwen 承担，且调用次数必须受控。

## 2.3 任务形式冲突

原论文偏开放域多跳 QA；AFAC2026 赛题四更像：

- 金融长文档选择题 / 判断题
- 常含表格、年份、金额、条件例外、监管措辞
- 输出必须是严格答案格式，而不是自由生成长答案

**结论**：LogicRAG 在这里应服务于 **证据定位与逐选项判定**，而不是追求开放式多跳问答表现。

## 2.4 工程约束冲突

当前仓库已经形成明确约束：

- 不改默认主路稳定性优先
- 要兼容 A 榜 `doc_ids` 代理召回评估与 B 榜盲搜
- 要保留现有 `Solver` / `Retriever` / variants 的可比性
- 要先做小样本与 smoke，再考虑 live run

**结论**：LogicRAG 不能以“大重构”落地，只能先以 staged rollout 逐步替换局部能力。

---

## 3. 可移植核心

在不违背项目约束的前提下，LogicRAG 有三类核心值得迁移：

### 3.1 可移植核心一：题目级子问题 DAG

可保留“题目 -> 子问题 -> 依赖边 -> 分层执行”的骨架，但节点文本改成适合 BM25 的短查询表达，例如：

- 先定位文档/章节
- 再定位指标/年份/主体
- 再检验选项中的数值、因果或条件约束

这比当前纯规则 query expansion 更适合复杂多条件题。

### 3.2 可移植核心二：按层检索与合并查询

同一层节点可以合并成 **unified lexical query**，减少重复检索；上游节点若已明确文档、年份、主体，下游节点可带着这些约束做更窄的 BM25 查询。

这部分与当前 `logic_lite_rrf` 的规则子查询天然兼容，是最容易先落地的一层。

### 3.3 可移植核心三：证据剪枝与滚动记忆

不是把所有 chunk 都送进 solver，而是每层只保留：

- 当前节点直接相关的证据片段
- 可供下游节点复用的结构化中间结论
- 必要的出处信息（doc_id / 标题 / 片段）

这对本项目尤其重要，因为 AFAC2026 的瓶颈之一就是 **token 成本与长上下文噪声**。

---

## 4. 建议适配路线

总体建议：**不做“原版 LogicRAG 移植”，只做“AFAC-LogicRAG-lite staged rollout”。**

> 补充口径（2026-06-16 更新）：此前仓库里保留的一些旧 `full evidence / adaptive evidence` 结果，只能作为历史实验记录；它们**不再作为当前 baseline，也不应作为可提交方案依据**。后续所有 ARS/LogicRAG stop/go 判断，都只应在当前无 embedding 主线内部进行，即与 `question_options`、`logic_lite_rrf`、`crag_lite` 等合法可比对象比较。

### 4.1 Stage 0：只做文档化与评估口径固化

目标：先把适配边界写清楚，避免后续实现偏题。

范围：

- 不改 Python 代码
- 明确“Qwen-only + BM25-only + auditable evidence”的红线
- 明确先做 retrieval-only 验证，再做 solver/live run

### 4.2 Stage 1：retrieval-first 的 `logicrag_qwen_rrf`

目标：先验证“Qwen 生成 DAG/子问题”是否真的能提升代理召回。

建议实现形态：

- 输入：题干 + 选项
- Qwen 输出：3-6 个子问题、依赖边、可裁剪节点
- 本地执行：DAG 校验、去重、拓扑分层、同层 query 合并
- 检索：仍使用 BM25 / 文档级 BM25 / RRF
- 下游：最终仍交给现有 solver，不改答题主链

进入条件：只有当代理召回优于 `logic_lite_rrf`，才继续向 solver 侧推进。

### 4.3 Stage 2：逐选项判定版 LogicRAG

目标：把 DAG 结构从“整题检索”推进到“选项级证据判断”。

建议做法：

- 多选题按选项拆成独立判定单元
- 公共上游节点共用（文档定位、主体识别、年份范围）
- 选项特异节点单独检索（数值、条件、例外条款）
- 让 solver 接收更干净的选项级证据包

这一阶段仍应避免多轮自反思，只做 **证据更精准的输入优化**。

### 4.4 Stage 3：滚动记忆与低置信回补

目标：在不显著增加成本的前提下，加入 LogicRAG 真正有辨识度的 pruning/memory 机制。

建议做法：

- 每层产出短摘要记忆：已确认事实 / 未解决空白 / 下层检索约束
- 对低置信节点触发 1 次补检索，而不是整题重跑
- 与 `crag_lite` 对接：让 CRAG 负责“是否值得补检索”，让 LogicRAG 负责“补检索什么”

### 4.5 Stage 4：受控接入 small live run

目标：只在小样本、低风险范围内把 LogicRAG 接入真实答题流程。

建议约束：

- 默认不开启，仅作为实验 variant
- 只在财报、多条件监管题、多选题等高收益子集启用
- 必须保留和 `question_options` / `crag_lite` 的对照运行

---

## 5. retrieval-only 测试计划

该阶段只回答一个问题：**Qwen 生成的动态结构，是否在不改变 solver 的情况下提升召回质量？**

## 5.1 对照对象

至少与以下 baseline 对比：

- `question_options`
- `logic_lite_rrf`
- `crag_lite`

若只和 `logic_lite_rrf` 比，容易误判“结构更复杂但不如默认主路”。

## 5.2 数据切分建议

按题型和领域拆看，不只看整体平均值：

- 财报 / 研报 / 保险 / 监管 / 合同
- 单选 / 多选 / 判断
- 需要跨年份比较 vs 不需要跨年份比较
- 需要逐选项验证 vs 可直接定位答案

## 5.3 重点指标

优先级建议如下：

1. **all_gold@10 / recall@10**：是否更完整召回关键文档
2. **mrr@10 / hit@1**：是否把关键文档排得更前
3. **平均查询数**：是否明显增加检索次数
4. **平均 token 成本**：Qwen 规划是否过贵
5. **失败类型分布**：是 DAG 错、query 错，还是 BM25 本身无解

## 5.4 观察方法

每次 retrieval-only 测试至少抽查以下内容：

- Qwen 规划出的子问题是否可解释
- DAG 是否存在无意义链条或重复节点
- 合并 query 后是否丢失关键金融术语
- 上游节点输出是否真的帮助下游收缩检索范围

## 5.5 通过标准

建议至少满足以下之一再进入下一阶段：

- 整体 `all_gold@10` 或 `recall@10` 相比 `logic_lite_rrf` 有稳定提升
- 在财报/多选/多条件题子集上有明确优势，且成本可控
- 没有明显拖累 `hit@1` 与 `mrr@10`

---

## 6. small live run 计划

small live run 的目标不是冲榜，而是回答：**LogicRAG 结构是否真的改善了最终答题链路，而不是只改善 doc_id 代理指标。**

## 6.1 启动条件

仅在 retrieval-only 达标后启动，且需满足：

- 子问题规划 JSON 稳定，无明显格式崩溃
- 平均每题新增 Qwen 调用次数可接受
- 不改默认主链，只加实验开关

## 6.2 样本范围

建议先跑小样本：

- 20 题 smoke：覆盖财报、监管、合同、多选各类
- 50 题 sample：观察答案质量、成本、异常率
- 与 `question_options` / `crag_lite` 同题对照

## 6.3 观察指标

live run 除召回外，还应看：

- 最终答案准确率/一致性
- 证据是否更聚焦、引用是否更可审计
- solver 输入长度是否下降
- 是否出现“规划正确但总结误判”的新错误类型
- 单题耗时是否超出现有可接受区间

## 6.4 风险控制

small live run 期间建议保留以下保险措施：

- 当 DAG 生成失败时，自动回退 `logic_lite_rrf` 或 `question_options`
- 当子问题数超过阈值时截断，避免爆 token
- 当低置信补检索触发后仍无提升时，不继续迭代

---

## 7. stop/go 阈值

为了避免“方法上看起来高级，但实际不划算”，建议在项目内明确 stop/go 门槛。

## 7.1 Go 条件

满足以下多数条件时，可继续推进：

- retrieval-only 相比 `logic_lite_rrf` 有稳定增益
- 至少在一个高价值子集（如财报多选）有明显优势
- live run 中最终答案质量不劣于 `question_options` / `crag_lite`
- 单题额外 token / 延迟仍在团队可接受范围内
- 失败案例可以通过 pruning、query merge、回退策略解释并修正

## 7.2 Stop 条件

出现以下任一情况，应停止扩大接入范围：

- 代理召回没有稳定提升，甚至持续劣于 `logic_lite_rrf`
- Qwen 规划噪声过大，子问题经常重复、空泛或与金融术语脱节
- live run 准确率无提升，但 token/耗时显著上升
- 需要大量 prompt 特判才能稳定，维护复杂度高于收益
- 错误主因来自 lexical retrieval 上限，而不是推理结构本身

## 7.3 建议决策口径

- **Go**：把 LogicRAG 作为实验性强分支，在高收益题型上继续扩展
- **Hold**：保留 retrieval-first 版本，仅用于特定子集或离线对照
- **Stop**：若收益不明显，则把经验沉淀回 `logic_lite_rrf` / `crag_lite`，不继续做 full-agent 化

---

## 8. 最终判断

本项目不适合照搬原版 LogicRAG，但**适合吸收其“动态子问题结构 + 分层检索 + 剪枝记忆”的核心思想**。

对 AFAC2026 来说，最现实的路线不是“全面替换现有 RAG”，而是：

1. 先把 LogicRAG 降维成 **Qwen 规划 + BM25/RRF 执行** 的 retrieval-first 变体；
2. 先证明代理召回与小样本 live run 有收益；
3. 再决定是否继续演进为带 rolling memory 的 solver-level LogicRAG。

一句话结论：**值得做，但只能 staged rollout，且必须先证明 retrieval-only 阶段就有增益。**
