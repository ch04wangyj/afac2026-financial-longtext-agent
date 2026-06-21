# V10 集合级证据复核与数值事实账本

> 日期：2026-06-21
> 状态：已实现、待官网提交验证
> 目标：针对 V9 的主要风险做局部增强，不再用更重的全链路 Agent 替换已验证的 claim-centric 主路。

## 1. 现状判断

历史榜单结果表明，增加完整 LogicRAG、全程 thinking 或更多查询轮次不等于准确率提升：V7 为 `39.6108`，V8 的自适应检索升至 `47.3224`，V9 的 claim-centric 检索进一步升至 `49.5701`。因此 V10 聚焦三个仍未闭环的问题：

1. Top-K 中存在同文档近重复 chunk，跨文档、跨条件证据覆盖不足。
2. 每个选项独立判断后直接拼接答案，不能可靠处理多选 exact-match、全称断言和证据冲突。
3. 财报、保险数值虽被检索到，最终判断仍容易混淆指标、年份、单位或依赖模型心算。

## 2. 最新研究与可纳入部分

| 研究 | 可复用结论 | 本项目决策 |
|---|---|---|
| [SURE-RAG](https://arxiv.org/abs/2605.03534) | 证据充分性是集合级属性，应聚合支持、反驳、不充分、覆盖与冲突 | 引入局部三态 verdict 校准和一次集合级答案复核；不引入额外 NLI 模型 |
| [Self-Correcting RAG](https://arxiv.org/abs/2604.10734) | 在固定预算下把上下文选择建模为多维覆盖/背包问题 | 实现确定性贪心边际收益选择，覆盖文档、术语、数值、日期和结构槽位；不接 MCTS |
| [VeNRA](https://arxiv.org/abs/2603.04663) | 金融数值推理应先形成带类型、可审计的事实账本 | 实现只读数值事实账本和 Decimal 单位规范化；不照搬其训练型 Sentinel |
| [FinLongDocAgent](https://arxiv.org/abs/2604.03664) | 长财报的关键瓶颈是跨表定位、计算和验证 | 保留“检索不足才补检索”的路线，并将表格/指标槽位纳入证据选择 |
| [FinAgent-RAG](https://arxiv.org/abs/2605.05409) | 自适应路由、可执行算术和自验证对金融 QA 有效 | 本轮先落地自验证和事实账本；模型生成 Python 暂不进入正式主路 |
| [DCRC](https://arxiv.org/abs/2605.31064) | 数值问答应把证据审计、结构化和执行视为整体编译过程 | 采用“证据 -> 事实账本 -> 最终复核”的轻量编译链，不引入训练阶段 |
| [RAG-X](https://arxiv.org/abs/2603.03541) | 仅看答案准确率会掩盖无依据命中，应分离检索与生成诊断 | 输出 selected chunks、missing slots、calibration tags、local/final answer 供审计 |
| [Query2doc](https://arxiv.org/abs/2303.07678) | LLM 伪文档可增强 BM25，但也可能引入未经证据支持的词 | 仅在 evidence insufficiency 时保留受控查询扩展，不作为默认首轮噪声源 |
| [PDF Parsing and Chunking for Financial QA](https://arxiv.org/abs/2604.12047) | PDF parser、chunk 策略和结构保真会直接影响金融 QA 正确率 | 下一阶段对财报 parser 与 row/header/year/unit chunk 做答案级消融 |
| [Facet-Level Evidence Tracing](https://arxiv.org/abs/2604.09174) | 相关 chunk 已召回仍可能因 facet 缺失或证据整合错误而答错 | 将 option 条件拆为 facet，并记录 facet x chunk 覆盖，不只看 doc recall |
| [MARDoc](https://arxiv.org/abs/2606.05749) | 结构化证据记忆比持续累积完整交互轨迹更能降低上下文噪声 | 延续事实账本/证据槽位，不保留完整 Agent 历史，也不照搬多模态多 Agent |
| [TechRAG](https://arxiv.org/abs/2606.01613) | evidence gate 和定向 retry 有价值，但其向量、跨编码器、知识图和多 Agent 链很重 | 只采用规则化 evidence gate；其 embedding/cross-encoder 路径违反本赛题边界 |

说明：以上 2026 年论文目前均为预印本，适合作为工程假设来源，不应把论文报告数字直接视为本赛题收益。

## 3. V10 实现结构

```text
claim query bundles
  -> sparse retrieval + RRF/document rescue
  -> slot-aware evidence set selection
  -> local support/refute/insufficient verdict
  -> citation/coverage calibration
  -> cross-option evidence selection
  -> deterministic numeric fact ledger
  -> one-shot set verification
  -> strict answer parser
```

核心文件：

- `agent/retrieve/evidence_selection.py`：固定字符预算下的覆盖选择和缺失槽位报告。
- `agent/retrieve/fusion.py`：在 chunk RRF 尾部救援高排名但被切片分散的目标文档代表证据。
- `agent/reasoning/claim_set_verifier.py`：引用合法性、关键槽位、全称断言文档覆盖和集合冲突校准。
- `agent/reasoning/fact_ledger.py`：数值、单位、年份和指标的确定性抽取与规范化。
- `agent/reasoning/prompts.py`：集合级 exact-match 复核协议。
- `agent/reasoning/solver.py`：仅在质量模式下编排上述能力。

## 4. 明确删除或暂缓的方向

1. **不把 GraphRAG 作为 A 榜主路**：题目已给 `doc_ids`，核心是文档内证据定位与集合判断，不是开放域实体图遍历。
2. **不恢复全题多轮 LogicRAG**：历史 V5/V7 已显示复杂度增加可能退分；仅对证据不足执行受控补检索。
3. **不使用模型置信度投票代替证据**：置信度只作排序信号，支持/反驳必须有合法证据引用。
4. **暂不执行模型生成代码**：先用 Decimal 账本消除单位和原值混乱；待有标注集后再评估受限 Calculator DSL。
5. **暂不引入 embedding 或额外 reranker**：维持赛题当前 sparse-first、无 embedding、可审计边界。

## 5. 验证口径

V10 不能用“模型输出看起来更合理”作为完成标准。合入前必须满足：

1. 全量单元/集成测试通过，覆盖文档救援、证据选择、引用校准、事实账本和集合复核。
2. 真实高风险样本包含 multi、tf、数值、比较、全称和跨文档题，保存 local answer 与 set-verified answer 差异。
3. 官网分数单独记录为 V10；若未超过 V9 `49.5701`，按题型和诊断标签做消融，不把 V10 自动升级为默认主线。

## 6. 下一轮优先级

1. 建立有标签的 20-30 题高风险 dev set，分别评估 retrieval recall、claim relation、set exact-match，而不是只看最终答案。
2. 财报表格增加 row/header/unit/year 结构化检索单元，减少整表 chunk 的指标错配。
3. 为数值账本增加受限运算 DSL（加减乘除、同比、占比），要求每个操作数绑定 fact id。
4. 仅对 `missing_slots` 或冲突触发 query refinement，并记录补检索是否真正新增关键证据。

## 7. 评分细则复核（2026-06-21）

官方页面：<https://tianchi.aliyun.com/competition/entrance/532486/information>。当前页面正文为动态加载；本次重新请求官网并与仓库内 2026-06-09 保存的 `tianchi-snapshot.md` 逐项交叉核对，评分口径如下：

```text
Accuracy = Correct / Total
TokenScore = max(0, min(1, (5,000,000 - TotalTokens) / 5,000,000))
FinalScore = 100 * Accuracy * (0.7 + 0.3 * TokenScore)
```

对开发优先级的直接影响：

1. `multi` 去重排序后与标准答案完全匹配，不给部分分。集合级复核和比较端点覆盖属于 P0，而不是可选解释层。
2. 空答案、非法字符、顺序错误均直接计错，因此 V10 增加非空恢复作答，禁止任意解析污染。
3. 全部 Qwen API 调用都计 Token；当前 8 题高风险样本共 `280,870` Token，不能把集合复核成本视为免费。
4. 文档离线解析不计 Token，因此财报表格行、条款、单位、年份和指标别名应尽量在离线/规则层完成。
5. A 组提供 `doc_ids`、B 组不提供。A 榜当前问题主要是文档内局部证据召回；B 榜还需独立验证文档级盲搜。
6. 显式规则禁止任何 embedding。官网推荐流程中的泛化“向量索引”描述不能覆盖这一明确禁令。

## 8. 本轮计划回顾与更新

### 已完成

- 对同伴最新 `main` 完成差异审查；其新增提交仅更新 README/输出布局，无运行时代码冲突。
- 首轮 V10 框架、入口修复、证据选择、引用校准、事实账本、集合复核和文档救援已实现。
- 全量测试曾达到 `221 passed, 1 xfailed, 2 xpassed`；后续 option-first/weighted RRF 改动仍需最终全量回归。
- 8 道高风险真实题完整跑通，无 API/格式异常；该集合无官方标签，不能据此宣称准确率。
- `fin_a_011` 的人工原文审计暴露并验证了指标术语差异：option-first、财报指标别名与 weighted RRF 能召回比亚迪年报第 11 页核心财务表。
- 两道跨公司财报题的最终真实调用结果为 `fin_a_005=ABD`、`fin_a_011=ACD`；逐项原文核对与数值关系一致，但它们仍属于人工审计结论，不冒充官方标签准确率。

### 当前 P0

1. 将 `fin_a_005/fin_a_011` 的跨公司指标修复扩展到全部财报题，检查营业收入、净利润、研发强度、经营现金流和分红五类指标。
2. 建立 20-30 题人工核验 dev set；每题保存标准候选答案、关键 chunk、facet 和计算式，用它做 answer-level 消融。
3. 官网提交 V10，只有真实得分超过 V9 `49.5701` 才升级为正式提交主线。

### 下一阶段 P1

1. 离线生成 table row 级记录：`doc/year/metric/value/unit/header/page`，优先解决财报整表检索错位。
2. 在事实账本上增加只允许四则运算、同比、占比、阈值比较的 DSL，每个操作数必须绑定 fact id。
3. 用 dev set 比较“逐项 + 集合复核”与“仅风险题集合复核”，在准确率无回退后再压缩 Token。
4. 为 B 榜单独评估 document BM25F 的 `all_gold@K`，不能把 A 榜已知 `doc_ids` 的收益外推到盲搜。

### 继续删除的方向

- 不引入 embedding、cross-encoder、非 Qwen 压缩模型或训练型 NLI。
- 不把 GraphRAG、MCTS、全程多 Agent、全题 Self-Consistency 作为主路。
- 不再用 doc hit rate 代替最终答案准确率；它只能诊断召回层。
