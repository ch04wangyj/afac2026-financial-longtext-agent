# Outputs Layout

本项目默认将运行产物写入 `outputs/`，但不再鼓励把所有文件直接散落在根目录。

## 目标

- 明确区分：测试输出、样本输出、A 榜 100 题正式输出
- 让 `answer.csv / evidence.json / token_usage.json / run_report.*` 自动跟随对应运行目录
- 让对比报告进入专门的 compare 目录
- 让历史 probe / loose artifacts 可以通过安全脚本清理

## 目录约定

```text
outputs/
  tests/
    smoke/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
    retrieval_compare/
      live/<timestamp>_<output_name>/
  samples/
    sample20/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
    sample40/
      dry/<timestamp>_<strategy>/
      live/<timestamp>_<strategy>/
    compare/
      <baseline>__vs__<candidate>/
  a100/
    full100/
      live/<timestamp>_<strategy>/
    compare/
      <baseline>__vs__<candidate>/
```

## 各脚本默认行为

### `scripts/03_run_questions.py`
- 全量 100 题真实运行：`outputs/a100/full100/live/...`
- 其他 limit / domain 子集：`outputs/tests/...`

### `scripts/05_compare_rag.py`
- 默认输出到：`outputs/tests/retrieval_compare/live/...`

### `scripts/06_smoke_by_domain.py`
- dry-run：`outputs/tests/smoke/dry/...`
- live：`outputs/tests/smoke/live/...`

### `scripts/07_run_sample.py`
- sample20 / sample40：`outputs/samples/<sampleN>/{dry|live}/...`
- 若实际题数为 100：归类为 `outputs/a100/full100/live/...`

### `scripts/04_make_submission.py`
- 默认把 `answer.csv / evidence.json / token_usage.json` 写回 `--results` 所在目录

### `scripts/08_report_results.py`
- 默认把 `run_report.md / run_report.json` 写回 `--results` 所在目录

### `scripts/09_compare_runs.py`
- sample 对比：`outputs/samples/compare/...`
- A100 对比：`outputs/a100/compare/...`

## 显式覆盖规则

如果设置了 `AFAC_OUTPUTS_DIR`，脚本会优先尊重该目录，而不是自动生成规范路径。

这保留了：
- 手工指定长任务输出目录
- watcher / resume / 批量实验脚本的兼容性

## Resume 行为

当脚本使用 `--resume` 且**没有**显式设置 `AFAC_OUTPUTS_DIR` 时，会在对应规范目录下寻找最近一次同策略 run 目录继续写入。

建议：
- 长任务（尤其 full A100）仍优先显式设置 `AFAC_OUTPUTS_DIR`
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

2. 明显属于历史 probe / 临时 compare 的目录
   - `qwen_plus_*`
   - `smoke_insurance_*`
   - `compare_*`

> 该脚本默认 `dry-run`，不会直接删除。

## Git 约定

`outputs/` 整体仍保持 `.gitignore`，原因：
- 体积大
- 运行频繁变化
- 不适合作为源码版本历史的一部分

代码提交应只包含：
- 输出路径逻辑
- 清理脚本
- 测试
- 文档
