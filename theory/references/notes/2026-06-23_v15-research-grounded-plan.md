# V15 优化计划：基于论文调研的算法重构

> 制定日期：2026-06-23
> 基线：V14 官网得分 68.69，Token 312,541，反推准确率约 70/100
> 旧 V15（Qwen-VL+版面深化）已失败：66.7267，人工核验不可靠
> 目标：85+ 分（准确率 87/100）

## 一、赛题硬约束（不可违反）

- **仅允许 Qwen 系列模型 API**（基准 Qwen3.6-plus），通过阿里云百炼或魔搭社区调用
- **禁止 embedding**、向量数据库、非 Qwen reranker、非 Qwen 模型
- **禁止修改基座模型参数**（不可微调、不可 LoRA）
- 文档离线解析阶段 Token 不计入最终统计
- 所有在线 Qwen 调用必须记录 usage
- 评分：FinalScore = 100 × Accuracy × (0.7 + 0.3 × TokenScore)，TokenBudget=5,000,000
- 多选题不设部分分；答案字母需去重排序、无分隔符

## 二、论文调研发现

### 2.1 FinAgent-RAG（arXiv 2605.05409，2026-05，GitHub 开源）

**最直接相关的论文。** Agentic RAG 框架，FinQA/ConvFinQA/TAT-QA 分别达 76.81%/78.46%/74.96%。

三大创新（赛题约束下可用部分）：
- **Program-of-Thought (PoT)** ✅：生成可执行受限 DSL 代码做精确算术，消除 88% 算术错误。Qwen 生成代码文本，本地确定性执行，不违反约束
- **自验证迭代检索** ✅：最多 3 轮检索-推理循环，REJECT 时自动 query refine 重检索。用 Qwen 做 verifier
- **自适应策略路由器** ✅：简单题用 CoT（便宜），复杂题用 PoT（贵），API 成本降 41.3%
- **对比式金融检索器** ❌：需要 embedding 训练硬负样本，赛题禁止 embedding

### 2.2 MimirRAG（arXiv 2605.25030，2026-05）

FinanceBench 达 89.3%。三大使能因素：
- **表格感知分块** ✅：保留表格结构而非拆成普通文本（V14 已部分实现）
- **元数据集成** ✅：每块附带 doc_type/section/page/table_id 等元数据（已有 extra_index_fields）
- **智能体工作流** ✅：查询规划 → 混合检索 → 验证 → 生成

### 2.3 DCRC（arXiv 2605.31064，KDD 2026）

解决数值幻觉。核心：**编译执行推理**——把查询和证据转为可验证可执行程序。V11 已有受限计算 DSL 雏形，可扩展。

### 2.4 Two-Phase Retrieval（arXiv 2605.20684）

解决"相似性-效用差距"。BM25 排名靠前的可能只是"话题相关"而非"决策有用"。第二阶段用 **LLM-as-a-Judge** 按"分析效用"重新排序——是 LLM 评分，不是 embedding reranker，合规。

### 2.5 MARDoc（arXiv 2606.05749，2026-06）

三智能体：Explorer（多粒度检索）→ Refiner（蒸馏结构化证据）→ Reflector（检查充分性+定向反馈）。**结构化记忆**替代不断增长的上下文，减少噪声。

## 三、V15 技术方案

### 核心改造：从"单次检索+模型心算+人工核验"到"迭代检索+程序化推理+模型自验证"

```
题目输入
    │
    ▼
┌──────────────┐
│ 查询分解器    │ → 子问题 [s1, s2, ...]
└──────┬───────┘
       │
┌──────▼───────┐
│ 自适应路由器  │ → simple: CoT / complex: PoT
└──┬───────┬───┘
   │       │
┌──▼──┐ ┌──▼──┐
│ CoT │ │ PoT │ (受限DSL确定性执行)
└──┬──┘ └──┬──┘
   └──┬───┘
┌──────▼───────┐
│ 自验证器      │ → ACCEPT / REJECT
│ (3项检查)    │    (Qwen做verifier)
└──────┬───────┘
  ACCEPT│  REJECT│
       │ ┌──────▼──────┐
       │ │ Query Refine│ → 重检索（最多3轮）
       │ └─────────────┘
       ▼
   最终答案（不依赖人工核验）
```

### 改造 1：Program-of-Thought 数值推理（解决算术错误）

**问题**：当前模型靠"心算"比较数值，V15旧版有 3 题 conf=0 因证据检索到但计算错误。

**方案**（借鉴 FinAgent-RAG PoT + DCRC + V11 受限 DSL）：
- 检测题干中的比较谓词（高于/低于/超过/同比/占比/增长率）
- 让 Qwen 生成受限 DSL 代码（仅 compare/difference/ratio/growth_rate）
- 确定性执行，不依赖模型心算
- V11 已有 `numeric_fact_ledger` 和受限计算 DSL，扩展为完整 PoT

**约束**：不修改基座模型，只让 Qwen 生成代码文本，本地确定性执行。不引入 embedding。

### 改造 2：自验证迭代检索（解决证据不完整）

**问题**：当前单次检索，conf=0 时直接放弃，3 题因此失分。

**方案**（借鉴 FinAgent-RAG 自验证 + MARDoc 结构化记忆）：
- 首轮检索后，Qwen 做 3 项验证：
  1. 证据是否覆盖所有选项
  2. 关键数值是否有原文支撑
  3. 推理链是否完整
- REJECT 时自动 query refine（换同义词、扩大范围、换检索路径）
- 最多 3 轮，每轮排除已检索过的证据
- 结构化记忆（借鉴 MARDoc）：每轮蒸馏证据而非累积原始上下文

**Token 预算**：每轮约 2-3K token，3 轮最多 +9K token/题。只对 conf<0.8 的题启动迭代（预估约 30 题），增量约 +90K token。

### 改造 3：自适应策略路由（优化 Token）

**问题**：当前全部 --no-thinking，复杂多选题推理深度不足。

**方案**（借鉴 FinAgent-RAG 自适应路由器）：
- **简单题（mcq + tf，35 题）**：no-thinking，快速低成本
- **复杂题（multi，65 题）**：启用 thinking，深度推理
- **数值比较题**：路由到 PoT，生成代码执行
- 预估 Token：thinking 增加 ~120K，PoT 增加 ~30K

### 改造 4：LLM-as-a-Judge 效用重排（解决相似性-效用差距）

**问题**：BM25 排名靠前的可能只是"话题相关"而非"答案有用"。

**方案**（借鉴 Two-Phase Retrieval，**不用 embedding**）：
- 第一阶段：BM25 检索 Top-30 候选（现有）
- 第二阶段：Qwen 做 judge，按"对回答此问题的有用性"重新排序
- 只取 Top-5 进入推理上下文
- **是 LLM 评分，不是 embedding reranker，合规**

### 改造 5：全领域 V14 layout 索引覆盖（扩大优化范围）

**问题**：60% 题目（insurance/regulatory/contracts）仍用 V13 旧语料。

**方案**（借鉴 MimirRAG 表格感知+元数据）：
- insurance：条款编号原子化，免责条款层级展开
- regulatory：法规条文按施行日期/义务/时限/措施分槽位
- contracts：募集说明书按发行规模/利率/期限/增信结构化
- 纯离线，零在线 Token 成本
- 复用 V14 layout_pdf.py，不引入新模型

## 四、分阶段实施

### 阶段 1：PoT 数值推理 + 全领域索引（3 天）

1. 实现 `agent/reasoning/pot_reasoner.py`：
   - 检测比较谓词
   - Qwen 生成受限 DSL 代码（compare/difference/ratio/growth_rate）
   - 确定性执行
   - 与 V14 的 numeric_fact_ledger 集成
2. 扩展 V14 layout 索引到 insurance/regulatory/contracts
3. 单元测试 + 3 题开发集门禁

### 阶段 2：自验证迭代检索（3 天）

1. 实现 `agent/reasoning/self_verifier.py`：
   - 3 项验证检查（Qwen 做 verifier）
   - Query refine 策略（同义词/扩大范围/换路径）
   - 结构化证据记忆
2. 集成到 precise_verifier
3. 对 conf<0.8 的题启动迭代

### 阶段 3：自适应路由 + LLM Judge 重排（2 天）

1. 实现策略路由器：按题型/metadata 路由 thinking/no-thinking/PoT
2. 实现 LLM-as-a-Judge 重排模块（Qwen 做 judge，不是 embedding）
3. Token 预算控制

### 阶段 4：全量运行 + 模型自验证融合 + 提交（2 天）

1. 100 题全量运行
2. **不依赖人工核验**——用模型自验证做接受/拒绝
3. 开发集 3/3 门禁
4. 官网提交验证

## 五、Token 预算

| 项目 | V14 | V15 预期 |
|---|---:|---:|
| 在线答题 Token | 312,541 | ~430,000 |
| 其中 no-thinking（35题） | ~80K | ~80K |
| 其中 thinking（65题） | 0 | ~200K |
| 其中 PoT 执行 | 0 | ~30K |
| 其中自验证迭代（30题×2轮） | 0 | ~120K |
| TokenScore | 0.9375 | ~0.914 |
| 离线 Token（不计费） | 0 | ~40K（V15 VL + 全领域解析）|

**若准确率 80/100**：得分 = 80 × (0.7 + 0.3 × 0.914) = **78.0**
**若准确率 85/100**：得分 = 85 × (0.7 + 0.3 × 0.914) = **82.9**
**若准确率 87/100**：得分 = 87 × (0.7 + 0.3 × 0.914) = **84.8** → 接近 85

**关键：Token 增加会降低 TokenScore，必须确保准确率提升足够大。**

## 六、关键约束（不可违反）

- **仅用 Qwen 系列模型**，不引入非 Qwen 模型
- **禁止 embedding**、向量数据库、非 Qwen reranker
- **不修改基座模型参数**（不可微调、不可 LoRA）
- **不依赖人工核验改答案**（旧 V15 教训）
- PoT 只用受限 DSL（compare/difference/ratio/growth_rate），不允许任意代码
- 自验证用 Qwen，不引入非 Qwen 模型
- LLM Judge 用 Qwen，不是 embedding reranker
- 开发集 3/3 硬门禁
- 在线 Token 控制在 500K 以内（TokenScore ≥ 0.90）

## 七、与旧 V15 失败的关键区别

| 维度 | 旧 V15（失败 66.73） | 新 V15 |
|---|---|---|
| 答案变化依据 | 人工核验"原文支撑" | 模型自验证 + PoT 确定性执行 |
| 检索策略 | 单次 BM25 | 迭代检索 + query refine |
| 推理模式 | 全部 no-thinking | 自适应路由（thinking + PoT） |
| 数值比较 | 模型心算 | 程序化执行 |
| 证据排序 | BM25 分数 | LLM-as-a-Judge 效用排序 |
| 覆盖范围 | 40 题（财报+研报） | 100 题（全领域） |
| 研究依据 | 无 | 6 篇 ArXiv 论文 + 1 GitHub 开源 |
