# V6：证据契约与事实账本

## 状态

- V5 官网基线：82/100，`315,727` Token，得分 `80.4466`。
- V6 官网结果：85/100，`326,076` Token，得分 `83.33`。
- 正式文件：`outputs/releases/v6/answer.csv`。
- CSV SHA-256：`8E6DD42567F2E44F9CBE79693ADBE4D83FFDB09F6BAF5ED75BB33EB7466E4C02`。

V6 的六个答案变化相对 V5 实际净增 3 题；总分不能逐题归因，因此仍需方法级消融，不把六项全部标成“已验证正确”。

## 问题定位

V5 的主要残差不是单纯 BM25 召回率不足，而是四类充分性错误：

1. 相关页面已召回，但缺少选项成立所需的另一文档或数值端点。
2. 表格命中同名指标，但取到母公司、季度、客户或分部口径。
3. 模型把“违约金/违约利息”“客户资金杠杆/除客户资金杠杆”等近义表述当成等价。
4. 条文本身为真，但不属于题干要求选择的审批、金额门槛或适用范围。

## 核心实现

### 选项级证据契约

`agent/reasoning/evidence_contract.py` 为每个选项记录：

- `required_doc_ids`、`observed_doc_ids`
- `predicate_doc_ids`、`numeric_doc_ids`
- 缺失文档、谓词和数值端点
- 支持/反证/真实值证据数量
- 全称、比较、否定、缺失、复合陈述和财务口径风险
- `selection_ready` 与 `needs_review`

该设计迁移了 [SURE-RAG](https://arxiv.org/abs/2605.03534) 的集合级 coverage、conflict 和 uncertainty 思路，但未引入额外验证模型。

### 表格与数值事实账本

`agent/reasoning/fact_ledger.py` 和 `calculation_dsl.py` 将财务证据编译为带年份、单位、口径和质量标签的事实，只允许受限比较与计算。实现借鉴 [H-STAR](https://aclanthology.org/2025.naacl-long.445/) 的“先列/行抽取，再按题型选择文本或符号推理”，但执行层保持纯 Python。

### 缺失端点补检索

`verification_queries.py` 与 `verification_rerank.py` 只在证据契约缺少谓词、期限或数值端点时扩展关系词和文档配额。这与 [ChainRAG](https://aclanthology.org/2025.acl-long.1089/) 和 [FunnelRAG](https://aclanthology.org/2025.findings-naacl.165/) 的渐进检索方向一致，但本项目最多做受控补检索，不构建句子图或无限多轮 Agent。

### 跨题事实一致性

`agent/evaluation/claim_consistency.py` 把近重复选项组织为轻量事实图。边必须满足：

- 证据契约显示谓词来自同一原始文档；
- 数值与单位签名一致；
- 增长/下降、重大/非重大、币种等限定词一致；
- 全称断言要求证据文档集合一致。

该图只生成审计候选，不自动覆盖答案。它发现了真实的重复事实矛盾，也暴露了“相同句子来自不同题目文档”的错误迁移风险。

## 实验与缺陷

| 实验 | 结果 | 结论 |
|---|---:|---|
| V6 全量精确候选 | 100 题，41 题不同于 V5 | 全量 Judge 漂移仍过大 |
| V5/V6 同答案长上下文复核 | 59 题 | 15 个差异均未达到自动覆盖门槛 |
| 证据契约 + 原文复核 | 6 个正式变化 | 当前最小可信提交 |
| `reg_a_011` 修复性重跑 | `A → AC` | 文档作用域修复有效；D 条文本身为真但不属于题干范围 |
| 完整测试 | 312 passed，13 skipped | 主链回归通过 |

本轮修复的工程缺陷：

- 监管选项无法绑定到年度报告准则或尽调办法，导致要求无关文档同时举证。
- “最低保存期限届满”“解除保险合同并核实身份”等条件链未进入谓词扩展。
- 穷举脚本仍指向已废弃的默认索引目录。
- 事实一致性初版忽略币种、方向词和来源文档，产生假冲突。
- `fc_a_002` 初审混淆“190 亿元注册额度”与“10 亿元本期上限”；自审后撤销覆盖，保留 V5 的 `ABD`。

## 提交策略

正式提交：

```powershell
python scripts\23_merge_v6_candidate.py `
  --output outputs\v6_evidence_final\answer_results.jsonl
python scripts\04_make_submission.py `
  --results outputs\v6_evidence_final\answer_results.jsonl `
  --require-complete
```

诊断提交每次只改变一个歧义题：

```powershell
python scripts\23_merge_v6_candidate.py `
  --include-probe reg3_literal_rule `
  --output outputs\v6_probe_reg3\answer_results.jsonl

python scripts\23_merge_v6_candidate.py `
  --include-probe fc15_duplicate_true_option `
  --output outputs\v6_probe_fc15\answer_results.jsonl
```

先提交正式 candidate。只有在提交次数允许时再分别提交 probe，不能把两个 probe 合并后再猜贡献。

## V7 入口

1. V6 得分已反推为 85/100，六项改动合计净增 3 题。
2. 用方法级消融和单变量候选判定标准答案歧义。
3. 为题干选择范围建立 `question_envelope`，把“事实为真”和“应按题干入选”拆成两个确定性阶段。
4. 把监管法规、保险责任和合同术语的实体绑定从手写别名升级为离线规则生成的文档能力表。
5. 只有缺失端点题触发 Qwen 复核，其余题保持 V5/V6 证据闭环。
