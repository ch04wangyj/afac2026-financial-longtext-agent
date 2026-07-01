# V13：可信约束下的无下行候选

## 输入边界

- 官网基线：V9，`88/100`，`327,052` Token，综合分 `86.2732`。
- 硬约束运行：哈希匹配的 V4-V12。
- 可共享快照：`evaluation_data/leaderboard/v4_v12_answers.csv`，仅保存题号和答案。
- 排除：V1/V2 快照与可信历史联合不可行；V3 的本地 Token 不能精确解释官网分数。
- 不固定任何仅由原文语义或模型共识推断的隐藏标签。

## 候选

| QID | V9 | V13 | 约束结论 |
|---|---:|---:|---|
| `reg_a_004` | AC | ABC | 旧答案必错；新答案为增益或双错 |
| `res_a_011` | ABC | ABCD | 旧答案必错；新答案为增益或双错 |

两题联合变化的全部可行正确题数为 `88/89/90`。单独修改任一题只能得到 `88/89`，
因此联合版本同时具有更高上界和不低于 V9 的数学下界。该区间依赖本地快照与官网
提交精确对应，仍须由下一次官网提交验证。

## 产物

- 提交：`outputs/v13_safe_gain/answer.csv`
- Token：`327,052`
- SHA-256：`7390FD5F15555F2AC519305016AE61950203F8293E8C01FFCA3D07622386AC08`

## 复现

```powershell
python scripts\35_rank_active_probes.py `
  --baseline v9 `
  --alternative "reg_a_004=AC,ABC" `
  --alternative "res_a_011=ABC,ABCD" `
  --output outputs\audits\v13_probe_ranking.json

python scripts\31_build_residual_candidate.py `
  --baseline outputs\v9_target90\answer_results.jsonl `
  --config configs\v13_safe_gain_reviews.json `
  --profile safe_gain `
  --baseline-correct 88 `
  --output outputs\v13_safe_gain\answer_results.jsonl

python scripts\04_make_submission.py `
  --results outputs\v13_safe_gain\answer_results.jsonl `
  --output-dir outputs\v13_safe_gain `
  --require-complete
```

## 下一步

官网返回 88、89、90 时分别对应三个剩余模式。只有获得该反馈后，才能继续归因
`reg_a_004` 与 `res_a_011` 的命中状态；在此之前不再扩展第三标签批量覆盖。
