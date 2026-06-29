# V1 财报指标行索引、受限计算与答案级门禁

> 日期：2026-06-21
> 状态：本地实现与小样本验证完成，待扩大 dev set 和官网提交

## 1. 为什么继续改检索单元

前置原型已能通过财报指标别名找回正确整页，但整页仍包含大量无关数字。集合复核能答对，不代表证据结构可靠。根据 [Empirical Evaluation of PDF Parsing and Chunking for Financial Question Answering with RAG](https://arxiv.org/abs/2604.12047)，parser 与 chunk 策略的结构保真会直接影响金融 QA；这与本地错误模式一致。

因此 V1 不增加新 Agent，而是把退化为文本的财务表编译为：

```text
doc_id + page + metric + header + year + raw_value + unit + parent_chunk_id
```

## 2. 实现

### 2.1 行级财报 chunk

- `agent/preprocess/financial_rows.py`：识别营业收入、归母净利润、研发投入、经营现金流、分红、每股收益等指标。
- 支持表头列映射、括号负数、`元/千元/百万元/万元/亿元/%`。
- 区分表格行和叙述行，避免把邻近表头错误继承给“某一时点/某一时段”金额。
- `scripts/13_augment_financial_metric_rows.py`：复用现有 chunks，避免重新解析 PDF。

当前产物：原有 chunks 上新增 `787` 个指标行，总计 `8,897` chunks。

### 2.2 事实账本与计算 DSL

- 行级 chunk 的 cells 直接进入事实账本，标记 `extraction_mode=financial_row`。
- 旧文本正则事实仍用于展示，但不允许进入自动计算。
- 白名单操作：`compare`、`difference`、`ratio`、`growth_rate`。
- 每个操作数必须绑定 fact id；跨主体比较要求相同 metric、year 和单位族；同比要求同主体、同 metric、同单位族且年份不同。
- 不执行任意表达式，不执行模型生成 Python。

这与 [DCRC](https://arxiv.org/abs/2605.31064) 的 compile-and-execute 可审计方向一致，但本项目只采用确定性轻量子集。

### 2.3 答案级 dev gate

新增 `devsets/answer_level_v1.jsonl` 和 `scripts/09_eval_answer_devset.py`：

- 多选按比赛规则排序去重后完全匹配。
- 缺失题按错误处理，禁止只统计成功样本。
- 同时检查 required docs 和最小关键 chunks。
- 每条标签记录 provenance 和当前题面版本。

## 3. 真实结果

| qid | 前置原型 | V1 | 原型 Token | V1 Token |
|---|---|---|---:|---:|
| `fin_a_005` | `ABD` | `ABD` | 35,347 | 26,160 |
| `fin_a_011` | `ACD` | `ACD` | 30,668 | 29,608 |

两题合计 Token 从 `66,015` 降至 `55,768`，下降约 `15.5%`。V1 最终证据中分别包含 6 和 4 个行级 chunk。

当前 dev gate：

```text
total=3
correct=3
accuracy=1.0
required_evidence_all_hit=3/3
```

该数字只表示三个已人工核验样例无回归，不能作为 A 榜准确率。

## 4. 数据版本问题

仓库 `tianchi-snapshot.md` 中旧版 `reg_a_014` 是复杂担保场景，标准答案为 `AC`；当前公开 A 组中的同 qid 已变成“股东大会职权及会议规则”，选项和 `doc_ids` 均不同，人工核验答案为 `ABD`。

结论：dev label 必须绑定题面哈希，不能只按 qid 复用旧标签。V1 已增加 `question_sha1`，题干、选项、题型或 `doc_ids` 变化都会使 strict gate 失败。

## 5. 下一步

1. 把 dev set 扩展到至少 20 题，优先覆盖 multi、跨文档、全称、否定、阈值和表格计算。
2. 为 dev case 增加更多关键证据锚点和计算式，区分答案正确但证据链错误的情况。
3. 对 20 个财报问题跑 row-index retrieval 消融，统计目标指标行进入 Top-K 的比例。
4. 仅在 dev exact-match 不下降时，将 row chunks 合并进正式 `processed_data/chunks.jsonl` 和主索引。
5. 官网提交 V1；没有真实榜分前不更新“最佳得分”。
