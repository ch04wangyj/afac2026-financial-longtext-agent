# Outputs Layout

本项目默认将运行产物写入 `outputs/`。

当前文档只描述**仍在维护的输出面**：
- 默认主线运行
- LogicRAG 保留实验线运行
- sample / smoke / A100 正式输出
- Docling 样本导出与分析输出

## 目标

- 明确区分：测试输出、样本输出、A 榜 100 题正式输出、Docling 样本输出
- 让 `answer.csv / evidence.json / token_usage.json / run_report.*` 自动跟随对应运行目录
- 让 loose artifacts 可以通过安全脚本清理

## 目录约定

```text
outputs/
  tests/
    <run_name>/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
    smoke/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
  samples/
    sampleN/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
  a100/
    full100/
      live/<timestamp>_<strategy>/
  docling_samples/
    <domain>/<doc_id>/...
    analysis_summary.json
```

## 各脚本默认行为

### `scripts/03_run_questions.py`
- 全量 100 题真实运行：`outputs/a100/full100/live/...`
- 其他 limit / domain 子集：`outputs/tests/<run_name>/{dry|live}/...`
  - 例如：`run_questions_limit2`

### `scripts/06_smoke_by_domain.py`
- dry-run：`outputs/tests/smoke/dry/...`
- live：`outputs/tests/smoke/live/...`

### `scripts/07_run_sample.py`
- sample20 / sample40 / sampleN：`outputs/samples/sampleN/{dry|live}/...`
- 若实际题数为 100：归类为 `outputs/a100/full100/live/...`

### `scripts/04_make_submission.py`
- 默认把 `answer.csv / evidence.json / token_usage.json` 写回 `--results` 所在目录

### `scripts/08_report_results.py`
- 默认把 `run_report.md / run_report.json` 写回 `--results` 所在目录

### `scripts/11_export_docling_samples.py`
- 输出到：`outputs/docling_samples/<domain>/<doc_id>/...`

### `scripts/12_analyze_docling_samples.py`
- 汇总输出：`outputs/docling_samples/analysis_summary.json`
- 单样本输出：`outputs/docling_samples/<domain>/<doc_id>/analysis.md`

## 显式覆盖规则

如果设置了 `AFAC_OUTPUTS_DIR`，脚本会优先尊重该目录，而不是自动生成规范路径。

适用场景：
- 手工指定长任务输出目录
- watcher / resume 兼容
- 单独隔离一次运行结果

## Resume 行为

当脚本使用 `--resume` 且**没有**显式设置 `AFAC_OUTPUTS_DIR` 时，会在对应规范目录下寻找最近一次同策略 run 目录继续写入。

建议：
- 长任务（尤其 full A100）优先显式设置 `AFAC_OUTPUTS_DIR`
- 短任务 / dry-run 可直接依赖默认规范路径

## Cleanup

先预览：

```bash
python scripts/10_cleanup_outputs.py --dry-run
```

确认后执行删除：

```bash
python scripts/10_cleanup_outputs.py --apply
```

保留关键目录：

```bash
python scripts/10_cleanup_outputs.py --dry-run \
  --keep-dir a100_logicrag_parallel_thinking_2026-06-18_090743 \
  --keep-dir a_full_logicrag_agent_20260617
```

## 当前清理策略

默认会清理两类对象：

1. `outputs/` 根目录散落的 loose files
   - `answer.csv`
   - `answer_results.jsonl`
   - `evidence.json`
   - `token_usage.json`
   - `run_report.md`
   - `run_report.json`

2. 明显属于历史临时输出的目录
   - `qwen_plus_*`
   - `smoke_insurance_*`
   - `compare_*`

> 该脚本默认 `dry-run`，不会直接删除。

## Git 约定

`outputs/` 整体保持 `.gitignore`，原因：
- 体积大
- 运行频繁变化
- 不适合作为源码版本历史的一部分

代码提交应只包含：
- 输出路径逻辑
- 清理脚本
- 测试
- 文档
