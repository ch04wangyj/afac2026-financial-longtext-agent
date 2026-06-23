# 版本得分变更记录

> 只记录两类信息：**真实提交得分**、**相对上一轮引入的关键改动**。

## 当前最佳
- **最佳版本**：V14 确定性 PDF 版面与表格增量索引
- **最佳得分**：**68.69**
- **相对 V13 提升**：`+2.4308`
- **准确率说明**：按本地留存 Token `312,541` 与官方公式反推约 **70 / 100**；本地 Token 与实际提交口径可能存在差异，该反推仅供参考

---

## V15（旧）｜2026-06-23 离线 Qwen-VL + 版面算法深化（失败）
- **得分**：**66.7267**
- **相对 V14 变化**：`-1.9633`
- **准确率**：反推约 **68 / 100**，较 V14 下降 2 题
- **失败原因**：5 个答案覆写中净损失 2 题（人工核验仍不可靠，部分"原文支撑"判断有误）
- **已回退**：所有答案覆写已清空，恢复 V14 状态
- **保留资产**：V15 代码（layout_pdf.py B1-B4、vl_table_extract.py、scripts 21-23）和 Qwen-VL 离线提取结果保留供后续实验复用
- **重命名**：旧 V15 计划笔记更名为 `2026-06-23_v15-old-vl-layout-deepening-failed.md`，新 V15 计划见 `2026-06-23_v15-research-grounded-plan.md`

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

## V8｜2026-06-21 LLM LogicRAG adaptive retrieval 版本
- **得分**：**47.3224**
- **相对上一轮变化**：`+7.7116`
- **相对历史最佳变化**：`+6.8209`
- **相对 V7 新引入的关键改动**：
  - 将非 multi 的 `logicrag_agent` 主路径升级为 LLM 主导的 adaptive retrieval loop：LLM 生成 rank query bundles、LLM sufficiency gate 判断证据是否足够，不足时由 LLM 重写下一轮检索方向。
  - 保留 A 榜 `doc_ids` 严格约束；本轮 adaptive rank metadata 记录 query、scope、sufficiency、rounds 与 exhausted 状态。
  - 完成 focused regression、非 multi 真实 smoke 与完整 A100 实跑；本地产物目录为 `outputs/adaptive_logicrag_a100_20260621_004712`，提交文件为 `answer.csv`。

---

## V9｜2026-06-21 claim-centric retrieval 版本
- **得分**：**49.5701**
- **相对上一轮变化**：`+2.2477`
- **相对 V8 新引入的关键改动**：
  - 将 `multi` 与 `mcq` 统一重构为 **claim-centric retrieval**：先按 option-level claim 建模，再共享 claim target / query bundles / sufficiency / refinement / verdict 流程。
  - 为 claim-centric 路径加入显式 token budget 控制：限制 claim query bundles、最多一轮 refinement、限制 verdict prompt 的 evidence 条数，并默认关闭 claim final compose。
  - 完成 focused regression、sample20 gate 与完整 A100 实跑；本地产物目录为 `outputs/claim_centric_a100_20260621_121023`，提交文件为 `answer.csv`。

---

## V10 candidate｜2026-06-21 集合级证据复核与数值事实账本
- **得分**：待官网提交验证，不使用本地样本结果代替榜单分数。
- **相对 V9 新引入的关键改动**：
  - 在 claim-centric 路径加入文档/实体/数值/日期/后果槽位感知的证据集合选择，减少相似 chunk 挤占上下文，并为 RRF 增加文档级代表 chunk 救援。
  - 修正 claim target 为 option-first，并加入财报指标披露名词典、逐文档查询和 weighted RRF，解决“归母净利润/归属于上市公司股东的净利润”等术语差异导致的比较端点漏召回。
  - 对局部 `support / refute / insufficient` verdict 校验证据编号、关键槽位和全称断言跨文档覆盖；多选和高风险单选增加一次集合级 exact-match 复核。
  - 增加确定性数值事实账本，抽取指标、年份、单位和规范化值，避免最终复核仅依赖模型心算。
  - 修复 `03_run_questions.py`、`07_run_sample.py` 中 A 榜质量模式函数名错误，并补充端到端与回归测试。
- **启用方式**：`python scripts\03_run_questions.py --a-board-quality`。

---

## V11｜2026-06-21 财报指标行索引与答案级门禁
- **得分**：**58.1679**。
- **相对 V9 变化**：`+8.5978`。
- **准确率**：根据官方公式和提交 Token 反推为 **62%**。
- **相对 V10 新引入的关键改动**：
  - 从现有财报 text chunks 确定性提取 `metric/year/value/unit/header`，生成 787 个短指标行 chunk，并接入 BM25F structured 字段。
  - 事实账本优先使用行级单元格列映射；增加只允许 `compare/difference/ratio/growth_rate` 的计算 DSL，禁止模型生成代码和跨单位计算。
  - 新增带当前题面、来源和关键证据的 answer-level dev set，以及 `09_eval_answer_devset.py` exact-match 门禁。
  - 两道财报真实回归保持 `fin_a_005=ABD`、`fin_a_011=ACD`；合计 Token 从 `66,015` 降至 `55,768`（约 `-15.5%`）。
  - 当前三题门禁为 `3/3`，仅用于回归，不代表官方 A 榜准确率。

---

## V12 candidate｜2026-06-22 文档级高召回验证与反证审计
- **得分**：**57.7848**。
- **准确率**：根据官方公式和 `2,292,333` Token 反推为 **67%**。
- **相对 V11**：准确率提升 5 个百分点，但综合分下降 `0.3831`，主要原因是 Token 增加约 126 万。
- **相对 V11 新引入的关键改动**：
  - 不再把 sparse Top-K 当作证据入口；每个选项对每个指定 `doc_id` 独立执行 BM25F 查询。
  - 增加非年份数值、完整日期和实体 exact sweep，并展开同页与相邻 chunk。
  - 使用 Qwen3.7-Max 对平衡证据集统一裁决，显式区分陈述真伪与题干选择规则。
  - 对 42 道首轮差异题执行独立反证审计，4 道回退旧答案，最终相对 V11 修改 38 题。
  - 修复港股“末期股息/建议派发”查询缺口，以及“末期方案”和“全年合计”混淆。
- **提交产物**：`outputs/submissions/v12_exhaustive_audit_20260622/answer.csv`。
- **Token**：`2,292,333`。
- **开发集**：`3/3`，仅作为回归门禁，不代表官网准确率。

---

## V13｜2026-06-22 原子谓词检索与保守融合
- **得分**：**66.2592**。
- **相对 V12 变化**：`+8.4744`。
- **准确率**：官网未返回原始正确题数；本地留存 `377,650` Token 与实际提交口径不一致，因此不强行反推。
- **相对 V12 新引入的关键改动**：
  - 将 8110 个旧父块重建为 38544 个可检索原子子块，父块不参与 BM25 排名，仅在代词/例外条件不完整时恢复局部上下文。
  - 修复小数误判条款、短法条被最小长度过滤、题干模板短语污染查询和普通年份被当成支持值的问题。
  - 查询拆成候选值支持与谓词真实值两路；财务行按指标、年份、表头和单元格完整度精排。
  - 每个选项最多 4 条证据，全题默认 12000 字；3 题开发门禁保持 `3/3`。
  - 全量无 thinking 单次运行 Token 为 `301,379`；保守融合及 thinking 审计后为 `377,650`。
  - 39 道弱证据变化回退 V12；逐条复核后最终仅保留 6 道有直接原文支撑的变化。
- **提交产物**：`outputs/submissions/v13_precise_reviewed_20260622/answer.csv`。
- **Token**：`377,650`；超过 V12 得分所需最低准确率约为 `59.13%`。
- **开发集**：`3/3`；V12 同三题审计版 `105,730` Token，V13 无 thinking 版 `8,839` Token。

---

## V14｜2026-06-23 确定性 PDF 版面与表格增量索引
- **得分**：**68.69**。
- **相对 V13 变化**：`+2.4308`。
- **准确率**：按本地留存 Token `312,541` 与官方公式反推约 **70 / 100**（较 V12 的 67 题多 3 题，较 V13 本地 Token 反推的约 68 题多约 2 题）。
- **相对 V13 新引入的关键改动**：
  - 读取 PDF 原生文字坐标和矢量线，恢复同一视觉行、双栏阅读块、有框表格与坐标对齐无线框表格；不调用 OCR、VLM、embedding 或非 Qwen 模型。
  - 将标题、单位、年度层级表头和跨页续表标记绑定到每个 `layout_table_row`，降低数值跨列、跨年和单位错配。
  - 作为 V13 原子语料的增量旁路，而非替换全文；24 个实际题目 PDF 共新增 32,663 块，其中结构化表格行 14,302，解析失败为 0。
  - 检索增加研报/财报指标别名、结构化表格加权和版面元数据透传；40 道财报/研报题重新运行。
  - 候选差异默认回退 V13，只有四道逐条原文复核记录可以覆盖；最终相对 V13 修改 `res_a_002`、`res_a_011` 两题。
- **提交产物**：`outputs/submissions/v14_layout_reviewed_20260622/answer.csv`。
- **Token**：`312,541`，相对 V13 本地底稿下降 `65,109`（约 `17.2%`）。
- **开发集**：`3/3`；提交文件 100 题完整、Token 汇总一致、UTF-8 BOM。

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
