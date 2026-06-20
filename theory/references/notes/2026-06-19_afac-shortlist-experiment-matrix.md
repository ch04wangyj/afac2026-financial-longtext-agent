# AFAC ranked shortlist and experiment matrix

## Purpose

This note turns the Phase 2 literature scan plus AFAC-fit analysis into a build order for this repo. It is intentionally implementation-facing: each option is framed as a code change against the current sparse-first A-board stack, not as a paper summary.

## Ranking summary

| rank | option | why it is this high |
|---:|---|---|
| 1 | BM25F-lite / field-aware sparse scoring | Lowest-risk retrieval lift: already has the right metadata (`title`, `clause_id`, `numbers`, `dates`, `chunk_type`) and can improve ranking quality without adding LLM calls. |
| 2 | T-RAG-lite / table row-block retrieval units | Most direct fix for financial-report questions, where current whole-table chunks are too coarse and often bury the decisive row. |
| 3 | Doc-scoped granularity routing | Very AFAC-specific: A-board already gives `doc_ids`, so the practical problem is choosing the right local unit (clause/paragraph/table) inside known docs. |
| 4 | Typed evidence-gap retry for uncertain options | Useful control-layer upgrade after retrieval structure improves; likely helps multi-select stability, but should sit on top of stronger base retrieval rather than substitute for it. |

## Experiment matrix

| rank | title | idea summary | exact AFAC fit | likely files to change | validation design | expected accuracy upside | expected token/cost risk |
|---:|---|---|---|---|---|---|---|
| 1 | BM25F-lite / field-aware sparse scoring | Replace the current post-fusion heuristic boost with a real field-aware sparse score path that separately rewards title/clause/number/date/chunk-type matches before final ranking. | AFAC A-board already constrains search by `doc_ids`, so better intra-doc ranking matters more than broader recall. This especially fits insurance/regulatory/contracts questions where clause ids, dates, and titles are decisive. | `agent/index/bm25.py`, `agent/retrieve/variants.py`, `agent/retrieve/retriever.py`, `agent/schemas.py`, `tests/test_structured_rag_variants.py`, `scripts/05_compare_rag.py` | 1) Add a new retriever variant beside `field_boosted_rrf`. 2) Run `python scripts/05_compare_rag.py --variants field_boosted_rrf <new_variant>` on A-board proxy retrieval metrics. 3) If metrics improve, run `python scripts/06_smoke_by_domain.py --per-domain 2` and a small `python scripts/07_run_sample.py --a-board-quality --sample-size 20` check to confirm no answer-quality regression. | Medium, broad-based upside. Best chance is better `hit@1`, `mrr@10`, and cleaner evidence ordering across all non-report domains. | Low. Mostly extra scoring logic and maybe slightly richer index metadata; no additional LLM calls. |
| 2 | T-RAG-lite / table row-block retrieval units | Re-index report tables as row/block evidence units instead of one whole-table chunk so retrieval can hit the exact financial line item, year pair, or metric row. | Financial reports are the most structurally table-heavy AFAC domain. Current `build_table_chunk()` stores a whole table body, which is often too long and too noisy for BM25 ranking and evidence compression. | `agent/preprocess/chunkers.py`, `agent/preprocess/docling_adapter.py`, `agent/index/build.py`, `agent/index/bm25.py`, `agent/retrieve/context.py`, `tests/test_docling_adapter.py`, `tests/test_context_expansion.py`, `tests/test_financial_report_calculator.py`, `scripts/02_build_index.py` | 1) Add row/block chunk generation behind a flag or parallel chunk type. 2) Rebuild indexes with `python scripts/02_build_index.py`. 3) Compare retrieval on financial-report-only questions with `python scripts/05_compare_rag.py --domains financial_reports`. 4) Then run `python scripts/06_smoke_by_domain.py --domains financial_reports --per-domain 3` and a focused `python scripts/07_run_sample.py --a-board-quality --qids ...` set containing report/table questions. | High for report-domain questions; likely strongest upside on `recall@10`, `all_gold@10`, and final multi-select evidence completeness where answers depend on a specific row. | Low to medium. Query-time token cost is near-zero, but index size and chunk count will grow, and context assembly may need tighter pack limits to avoid flooding evidence with adjacent rows. |
| 3 | Doc-scoped granularity routing | Keep doc restriction fixed, then route retrieval/context assembly by local structure: clause-first for regulatory/insurance, paragraph-first for prose, table-first for report questions, instead of always treating chunks uniformly. | This is a direct adaptation of SmartChunk to AFAC: not “which document do I need?” but “which structure inside the provided documents is most likely to carry the answer?” It matches the current A-board operating mode exactly. | `agent/retrieve/retriever.py`, `agent/retrieve/context.py`, `agent/retrieve/expansion.py`, `agent/retrieve/targets.py`, `agent/reasoning/logicrag.py`, `agent/reasoning/solver.py`, `agent/runtime/logicrag_config.py`, `tests/test_context_expansion.py`, `tests/test_logicrag_solver.py` | 1) Introduce simple routing rules keyed by question/domain signals and chunk metadata. 2) Evaluate against the current `logicrag_agent` / `multi_logicrag` path on mixed-domain samples. 3) Use `python scripts/06_smoke_by_domain.py --per-domain 2` to ensure each domain still gets evidence. 4) Use `python scripts/07_run_sample.py --a-board-quality --sample-size 20` to check end-to-end stability. | Medium. Biggest likely gains are fewer wasted evidence slots and better clause-vs-table placement, especially on A-board multi-doc questions. | Low to medium. Routing itself is cheap, but bad rules can suppress good evidence and indirectly increase retry frequency. |
| 4 | Typed evidence-gap retry for uncertain options | Replace generic “retry when uncertain” behavior with explicit gap types such as missing doc coverage, missing numeric evidence, missing table evidence, or contradictory evidence; each gap triggers one targeted re-query path. | AFAC multi-select mistakes often come from one missing option-specific evidence slice, not from total retrieval failure. The current retry path already exists, so the change is about making it more selective and less wasteful. | `agent/reasoning/solver.py`, `agent/reasoning/multi_logicrag.py`, `agent/retrieve/coverage.py`, `agent/retrieve/query.py`, `agent/runtime/logicrag_config.py`, `tests/test_logicrag_solver.py`, `tests/test_logicrag_config.py` | 1) Log gap categories in multi-option runs. 2) Compare retry rate, evidence coverage, and answer changes on a fixed A-board sample using `python scripts/07_run_sample.py --a-board-quality --sample-size 20`. 3) Inspect result deltas with `scripts/08_report_results.py` and `scripts/09_compare_runs.py` if needed. | Medium, mostly for multi-select robustness after top-3 retrieval changes are in place. It is more likely to rescue borderline questions than to lift the whole benchmark by itself. | Medium. This is the first option here that can materially add LLM and retrieval work per question if triggers are too loose. |

## Implementation notes per option

### Rank 1: BM25F-lite / field-aware sparse scoring

Recommended implementation shape:
- Keep the existing chunk corpus and tokenizer pipeline.
- Move from `_field_boost()` in `agent/retrieve/variants.py` toward a structured score composition in which title/clause/numbers/dates/chunk type contribute before final ordering, not only as a tiny heuristic after RRF.
- Preserve the current `field_boosted_rrf` variant as a baseline; add a separate experimental variant so `scripts/05_compare_rag.py` can compare them cleanly.

Why this is the best first experiment:
- It uses metadata the repo already stores in `Chunk` and `RetrievalResult`.
- It avoids index rebuild risk if implemented query-side first.
- It should improve ranking without increasing answer-time token usage.

What success should look like:
- Proxy retrieval: `hit@1` and `mrr@10` go up without hurting `all_gold@10`.
- Smoke runs: evidence text becomes more clause-specific and less generic.

### Rank 2: T-RAG-lite / table row-block retrieval units

Recommended implementation shape:
- Extend preprocessing so a parsed table can emit row/block chunks in addition to the current whole-table chunk.
- Add metadata such as parent table caption, row header, page, and maybe local row ordinal so downstream context expansion can merge related rows back together.
- Keep a reversible rollout: either dual-index both whole-table and row-block chunks, or add a config flag so you can A/B test without destroying the current path.

Why it is not rank 1 despite high upside:
- It requires index-time work, re-indexing, and more moving parts than BM25F-lite.
- It is highly likely to help reports, but less likely to move other domains.

What success should look like:
- Financial-report retrieval pulls the correct row-level evidence into top-10 more often.
- Evidence packs carry fewer irrelevant rows while preserving page/table context.

### Rank 3: Doc-scoped granularity routing

Recommended implementation shape:
- Add lightweight routing rules based on question/domain cues and retrieved metadata, not a planner-heavy agent.
- Use existing signals already present in the code: `chunk_type`, `clause_id`, `section`, page adjacency, and doc coverage.
- Treat this as a retrieval-and-context policy layer over known `doc_ids`, not as a new doc discovery stage.

Why it is third:
- It matches AFAC well, but the benefit depends on the retrieval base already surfacing the right anchor chunks.
- If BM25F-lite and row-block indexing are weak, routing has little good material to route.

What success should look like:
- Better use of evidence budget per question.
- Fewer cases where table questions get paragraph-heavy evidence or clause questions get table-heavy evidence.

### Rank 4: Typed evidence-gap retry

Recommended implementation shape:
- Reuse the existing multi-option retry path in `Solver._solve_multi_logicrag()`.
- Split “uncertain option” into a small finite set of retry reasons: missing doc coverage, missing table evidence, missing numeric/date support, contradiction between support/refute evidence, and low top-candidate concentration.
- Cap to one extra retry path per option and log trigger type so sample runs reveal whether the trigger is actually useful.

Why it is fourth:
- It improves control flow more than retrieval fundamentals.
- If base retrieval remains coarse, smarter retries will just spend more tokens searching the same weak space.

What success should look like:
- Similar or slightly lower retry count, but better answer flips on previously ambiguous multi-select options.
- More interpretable run logs because each retry has an explicit reason.

## Suggested build order

1. Build BM25F-lite first.
2. Build table row/block retrieval second.
3. Layer doc-scoped granularity routing on top of the stronger retrieval substrate.
4. Add typed evidence-gap retry last as a control policy, not as the main retrieval fix.

## Minimal experiment plan for the next engineering cycle

1. Retrieval proxy baseline
   - `python scripts/05_compare_rag.py --variants field_boosted_rrf logicrag_qwen_rrf`
2. Rank-1 experiment
   - add BM25F-lite variant and rerun `05_compare_rag.py`
3. Rank-2 experiment
   - rebuild index with row/block chunks and rerun `05_compare_rag.py --domains financial_reports`
4. End-to-end smoke
   - `python scripts/06_smoke_by_domain.py --per-domain 2`
5. Sample quality check
   - `python scripts/07_run_sample.py --a-board-quality --sample-size 20`

## Bottom line

If only one change ships next, it should be BM25F-lite because it is the cheapest broad retrieval upgrade.
If two changes ship, add T-RAG-lite row-block indexing immediately after it because that is the clearest fix for AFAC financial-report questions.
If three changes ship, make the third one doc-scoped granularity routing, not heavier agentization.
Typed evidence-gap retry is worth doing, but only after the retrieval substrate becomes more structure-aware.