# 参考文献与资料目录

本目录用于保存 AFAC2026 赛题四项目的参考资料，按用途分层管理：

- `official/`：官方资料、比赛页面导出的 PDF、规则截图、数据说明等
- `papers/`：论文、技术报告、RAG/长文本 Agent/压缩相关外部文献
- `notes/`：本项目自己编写的读书笔记、综述、摘录、对比表

约定：

1. `official/` 与 `papers/` 下的原始大文件默认不纳入 git；目录内保留 `.gitkeep` 以固定结构。
2. `notes/` 下的 Markdown 笔记可以正常纳入 git。
3. 若下载了新的官方说明，优先在文件名中带日期，例如：`2026-06-15_tianchi_rule_snapshot.pdf`。
4. 若引用某篇论文，建议在 `notes/` 中补一份同名摘要，记录其与本项目的关系、可借鉴点和限制。

当前相关项目文档仍保留在上层 `theory/` 目录；本目录更适合收纳后续新增的外部参考资料。

当前主线研究笔记：

- `notes/2026-06-28_v16-structure-selected-truth.md`：从 V14 的 68.6873 分出发，对 PageIndex、LongRefiner、BookRAG、表格深层结构编码和 query-generation table retrieval 做赛题适配与 V16 实现记录。
