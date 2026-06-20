# 版本得分变更记录

> 只记录两类信息：**真实提交得分**、**相对上一轮引入的关键改动**。

## 当前最佳
- **最佳版本**：2026-06-18 A-board quality 版本
- **最佳得分**：**40.5015**
- **相对历史最佳提升**：`+0.5513`

---

## V1｜2026-06-17 上午
- **得分**：**39.2775**
- **相对上一轮改动**：
  - 首条记录，作为历史基线。

---

## V2｜2026-06-17 晚上
- **得分**：**39.9502**
- **相对上一轮变化**：`+0.6727`
- **相对 V1 新引入的关键改动**：
  - 引入并正式全量运行 **自适应结构推理 / `logicrag_agent`**。
  - 重跑全量预处理与索引，使用 `68` 文档 / `14445` chunks 的正式产物。
  - 增加 Docling 内存治理与运行防护，避免长任务重叠和预处理阶段内存堆积。
  - 完成 100 题正式答题链路，并生成新的 `outputs/answer.csv` 提交结果。

---

## V3｜2026-06-18 本轮并发 + thinking 版本
- **得分**：**30.0709**
- **相对上一轮变化**：`-9.8793`
- **相对 V2 新引入的关键改动**：
  - 将 LogicRAG 关键 Qwen 调用统一切到 thinking 开启路径。
  - 引入 YAML 运行时配置中心化，增加 `question_workers / qwen_workers / qwen_request_limit / bm25_workers`。
  - 增加题级并发、rank 内 BM25 并发、多选逐项并发，以及确定顺序的并发安全结果聚合。
  - 完成新的完整 A100 实跑，输出目录为 `outputs/a100_logicrag_parallel_thinking_2026-06-18_090743`。

---

## V4｜2026-06-18 A-board quality 版本
- **得分**：**40.5015**
- **相对上一轮变化**：`+10.4306`
- **相对 V3 新引入的关键改动**：
  - 引入 A-board quality runtime mode，并在运行脚本中增加 `--a-board-quality` 开关。
  - 将 `coverage gate + option matrix + financial calculator metadata + domain coverage facets` 接入正式答题链路。
  - 增加 answer delta diagnostics 与更细粒度 run report issue tags，用于运行后诊断。
  - 完成新的完整 A100 实跑，输出目录为 `outputs/phase7_a100_a_board_quality_2026-06-18_212102`。

---

## V5｜2026-06-19 检索优化版本
- **得分**：**38.0525**
- **相对上一轮变化**：`-2.4490`
- **相对 V4 新引入的关键改动**：
  - 将 `--a-board-quality` 主路切换为 **LogicRAG-first**，关闭默认 `option matrix` / `multi-option judgement`，保持 A 榜 `doc_ids` 严格约束。
  - 为 LogicRAG 增加结构化 retrieval target、`question+options` seed query、保守 sparse feedback query 与短 hypothetical sparse query。
  - 新增 lexical / structural rerank，以及基于 clause / section / page / neighbor 的 deterministic context expansion / evidence pack。
  - 完成新的完整 A100 实跑，输出目录为 `outputs/logicrag_retrieval_opt_a100_2026-06-19`，并生成 `results.csv`。

---

## V6｜2026-06-19 多选强化 LogicRAG 版本
- **得分**：**39.9964**
- **相对上一轮变化**：`+1.9439`
- **相对 V5 新引入的关键改动**：
  - 为 `multi` 题型新增 **multi_logicrag** 主路，改为逐选项检索 / 逐选项 verdict / 程序化组装最终多选答案。
  - 对每个不确定选项强制执行一轮 **扩 query 集 + 扩检索空间 + 再分析**，而不是仅做整题 compose。
  - 修复 `{"answer":""}` 被 `reason` 中 A/B/C/D 污染成多选答案的解析问题，并增加对应回归测试。
  - 完成新的完整 A100 实跑，输出目录为 `outputs/multi_logicrag_a100_2026-06-19_114306`，并生成 `results.csv`。

---

## V7｜2026-06-19 第一阶段 paper-faithful LogicRAG 验证版本
- **得分**：**39.6108**
- **相对上一轮变化**：`-0.3856`
- **相对 V6 新引入的关键改动**：
  - 按第一阶段目标，将 LogicRAG 主路进一步收敛到 **paper-faithful** 执行语义，补齐 planner / DAG contract、rank-wise retrieval linearization 与 pruning 行为约束。
  - 将运行时配置、prompt contract 与测试约束显式化，新增 Phase 1 hardening 所需的 runtime defaults、thinking budget hierarchy 与对应回归测试。
  - 完成第一阶段 validation：跑通 focused regression、sample20 与新的完整 A100，本地产物目录为 `outputs/phase1_validation_a100_2026-06-19_160310`。

---

## 后续追加模板

```markdown
## Vn｜YYYY-MM-DD 时段
- 得分：
- 相对上一轮变化：
- 相对上一轮新引入的关键改动：
  - 
  - 
```
