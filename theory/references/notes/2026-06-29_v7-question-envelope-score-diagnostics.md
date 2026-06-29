# V7 题干范围门禁与分数诊断

## 结论

V6 的 `83.33` 分和 `326,076` Token 对应 `85/100`，不是估计区间。六个答案
变化相对 V5 合计净增 3 题，但仅凭一个总分不能逐题归因。

## 负向实验

把每个选项都拆成 `fact_truth/applicable/selected` 后，全量运行产生 42 个答案
变化和 `485,509` Token。模型将证据不足、未覆盖整个题干和不属于题干集合混为一谈，
导致大量假阴性。该版本不进入提交链。

## 收缩设计

范围门禁只对明确询问集合归属的题目启用，例如：

- 哪些情形需要内部审批或满足金额门槛；
- 哪些产品可以赔付；
- 哪些情形属于免责范围。

普通事实题仍输出 V6 的 `truth`。收缩后 6 道触发题全部复现 V6 答案。

## 工程产物

- `agent/reasoning/question_envelope.py`
- `agent/evaluation/score_diagnostics.py`
- `scripts/25_build_v7_ablation_matrix.py`
- `scripts/26_merge_v7_targeted_candidate.py`
- `configs/v7_targeted_reviews.json`

## 研究映射

- SURE-RAG：证据充分性是集合级属性，相关证据不等于可支持答案。
- S2G-RAG：显式判断证据缺口，再把缺口映射为受控补检索。
- 本项目迁移两者的结构化状态，不接入额外 verifier、embedding 或无限多轮 Agent。
