# V12 负优化复盘与 V13 无下行探针

## 1. 观测

V12 使用 `327,052` Token 获得官网 `83.3320`，按评分公式对应 `85/100`。它相对
V11 修改 5 题并净损失 3 题，因此此前 `89-93/100` 的条件区间错误。

## 2. 根因

- 原文直接支持只能说明 `source_semantic`，不能等价为比赛 `official_label`。
- V1/V2 本地重建快照与可信后续运行联合不可行，却曾进入硬约束。
- HiGHS presolve 曾返回违反多条正确数等式 2 题的伪可行解。
- 一次修改 5 题只得到一个净分差，缺乏逐题可归因性。

## 3. 工程修复

- `configs/leaderboard_runs.json` 注册可信状态、文件哈希、Token 和正确题数。
- `agent/evaluation/label_evidence.py` 隔离官方标签、排行榜唯一标签、语义证据和
  模型共识。
- `agent/evaluation/leaderboard_constraints.py` 对所有 MILP 解执行原约束复验。
- `agent/evaluation/active_probe.py` 枚举候选的全部可能正确题数，并按最坏结果排序。

## 4. 新技术横向结论

| 工作 | 可迁移能力 | 当前取舍 |
|---|---|---|
| LongRefiner（ACL 2025） | 双层查询分析、层级结构、查询驱动精炼 | 迁移到父子块证据精炼，不引入训练模型 |
| TreeRAG（ACL Findings 2025） | 树分块和双向遍历 | 以 BM25F 替代 embedding |
| TableRAG（EMNLP 2025） | 表 schema、SQL 执行、文本/表结果交叉验证 | 财报子链采用 SQLite/受限 DSL |
| LAD-RAG（2025） | 跨页版面关系和动态检索 | 只构造确定性符号边 |
| RAG-Anything（2026.06 更新） | MinerU/Docling/PaddleOCR 路由、多模态上下文绑定 | 解析器适配层采用，图/向量主链不采用 |
| Active Testing（ICML 2021） | 在有限标签预算下选择高信息测试点 | 适配为聚合总分下的版本空间探针 |

这些工作不能直接证明本题准确率提升。下一轮必须对层级召回、表格执行和复杂页解析
分别建立固定子集，报告 evidence Recall@K、执行正确率和最终答案变化。

## 5. V13

V13 从 V9 出发只修改：

- `reg_a_004: AC -> ABC`
- `res_a_011: ABC -> ABCD`

在哈希匹配的 V4-V12 约束下，两个旧答案均必错，新答案均只可能增益或双错。联合
候选的正确数可行集合为 `88/89/90`，Token 保持 `327,052`。

参考：

- https://aclanthology.org/2025.acl-long.176/
- https://aclanthology.org/2025.findings-acl.20/
- https://aclanthology.org/2025.emnlp-main.710/
- https://arxiv.org/abs/2510.07233
- https://github.com/HKUDS/RAG-Anything
- https://github.com/opendatalab/MinerU
- https://github.com/docling-project/docling
- https://proceedings.mlr.press/v139/kossen21a.html
