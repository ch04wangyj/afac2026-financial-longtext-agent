# Phase 2 convergence order: baseline-preserving retrieval upgrades

## Purpose

This note turns the stabilized Phase 1 baseline plus the Phase 2 shortlist into one concrete execution order for the AFAC repo. It is planning-only: no code changes are proposed here beyond naming the exact files, validation gates, and the Kanban card graph that should execute next.

Primary inputs:
- `theory/references/notes/2026-06-19_phase1-baseline-report.md`
- `theory/references/notes/2026-06-19_afac-shortlist-experiment-matrix.md`
- `agent/retrieve/variants.py`
- `agent/index/bm25.py`
- `agent/preprocess/chunkers.py`
- `agent/retrieve/retriever.py`
- `scripts/05_compare_rag.py`
- `scripts/06_smoke_by_domain.py`
- `scripts/07_run_sample.py`

## Decision summary

Priority #1 retrieval improvement: `BM25F-lite / field-aware sparse scoring`

Optional priority #2 fallback: `T-RAG-lite / table row-block retrieval units`

Baseline rule for all Phase 2 work:
- Preserve the Phase 1 planner/rank-memory/prompt/runtime contract and evaluate deltas against V7 (`39.6108`), not against pre-alignment variants.
- Treat retrieval substrate quality as the variable under test; do not reopen the whole LogicRAG planner/solver contract unless validation shows retrieval changes cannot move the target error class.

## Why BM25F-lite is first

1. Lowest implementation risk against the current repo.
   - The live code already stores the metadata BM25F-lite needs: `title`, `clause_id`, `numbers`, `dates`, and `chunk_type` in `agent/index/bm25.py` and `agent/schemas.py`.
   - A prototype path already exists in `agent/retrieve/variants.py` as `_field_boost()` plus the `field_boosted_rrf` variant, so Phase 2 can evolve a known path instead of introducing a new subsystem first.

2. Broadest likely upside per unit of change.
   - It targets the main unresolved Phase 1 reality: sparse retrieval quality inside known AFAC `doc_ids`.
   - It can improve clause-heavy insurance/regulatory/contracts ranking and numeric/date-sensitive questions without adding extra LLM calls.

3. Clean validation surface.
   - It can be judged first on retrieval-only proxy metrics with `scripts/05_compare_rag.py`, then on real solver behavior with `scripts/06_smoke_by_domain.py` and `scripts/07_run_sample.py`.
   - It does not require re-indexing, so an unsuccessful first experiment is cheap to revert or isolate.

## Why T-RAG-lite is second, not first

1. It is likely the strongest financial-report-specific upside, but it is operationally heavier.
   - `agent/preprocess/chunkers.py` currently emits one whole-table chunk via `build_table_chunk()`, so row/block retrieval requires chunk-shape changes and index rebuilds.
   - That means touching preprocessing, indexing, retrieval, and context assembly rather than mostly query-side scoring.

2. Its benefit is structurally narrower.
   - It is the clearest fix for `financial_reports`, but less likely than BM25F-lite to improve the non-report domains that still dominate broad A100 behavior.

3. It depends on a good anchor-selection substrate.
   - Even row-level evidence is less useful if the sparse ranker is still weak at surfacing the right table/table-region in the first place.

## Why the other shortlisted options are not first

### Doc-scoped granularity routing is not first
- It is a policy layer on top of retrieved anchors, not a direct retrieval-quality fix.
- `agent/retrieve/retriever.py` currently retrieves uniformly; routing before BM25F-lite/T-RAG-lite risks optimizing the use of weak anchors instead of improving anchor quality.
- It should follow stronger sparse ranking and better table granularity, not precede them.

### Typed evidence-gap retry is not first
- It improves control flow after uncertainty has already appeared.
- It can materially add retrieval and LLM work per question, which is the wrong first move while the baseline still has unresolved substrate issues.
- If base retrieval remains coarse, smarter retries mostly spend more budget searching the same weak evidence space.

## Exact files expected to change

### Priority #1: BM25F-lite / field-aware sparse scoring

Implementation files:
- `agent/retrieve/variants.py`
- `agent/index/bm25.py`
- `agent/retrieve/retriever.py`
- `agent/schemas.py`

Validation and support files:
- `tests/test_retrieval_variants.py`
- `tests/test_structured_rag_variants.py`
- `tests/test_logicrag_retrieval.py`
- `scripts/05_compare_rag.py`
- `scripts/06_smoke_by_domain.py`
- `scripts/07_run_sample.py`

What should change in those files:
- `agent/retrieve/variants.py`: add a separate experimental variant beyond `field_boosted_rrf`; move from post-fusion heuristic boosts toward explicit field-aware score composition and score breakdown logging.
- `agent/index/bm25.py`: expose richer per-result metadata or score hooks needed by a real field-aware sparse score path.
- `agent/retrieve/retriever.py`: if the variant proves out, wire the field-aware sparse scorer into the live retrieval path used by `logicrag_agent` instead of leaving it script-only.
- `agent/schemas.py`: only if needed for structured metadata/debug fields on retrieval outputs.
- tests/scripts: cover variant registration, ranking behavior, and report labels so the experiment is reproducible.

### Optional priority #2: T-RAG-lite / table row-block retrieval units

Implementation files:
- `agent/preprocess/chunkers.py`
- `agent/preprocess/docling_adapter.py`
- `agent/index/build.py`
- `agent/index/bm25.py`
- `agent/retrieve/context.py`

Validation and support files:
- `tests/test_docling_adapter.py`
- `tests/test_context_expansion.py`
- `tests/test_financial_report_calculator.py`
- `scripts/02_build_index.py`
- `scripts/05_compare_rag.py`
- `scripts/06_smoke_by_domain.py`
- `scripts/07_run_sample.py`

What should change in those files:
- `agent/preprocess/chunkers.py`: emit row/block table chunks in addition to or behind a flag for current whole-table chunks.
- `agent/preprocess/docling_adapter.py`: preserve row/caption/page metadata needed to reconstruct row-level evidence meaningfully.
- `agent/index/build.py` and `agent/index/bm25.py`: index the new chunk type and keep metadata searchable.
- `agent/retrieve/context.py`: merge neighboring rows/table context back into evidence packs so row retrieval does not lose interpretability.
- tests/scripts: verify row-level chunk generation, context pack behavior, and focused report-domain evaluation.

## Validation gates

Validation Gate A: retrieval-only proxy for Priority #1
- Command: `python scripts/05_compare_rag.py --variants field_boosted_rrf <new_bm25f_variant>`
- Required outcome:
  - `ALL` aggregate `hit@1` or `mrr@10` improves over `field_boosted_rrf`
  - `all_gold@10` does not regress materially
  - Report must clearly remain a proxy retrieval metric, not a benchmark score claim

Validation Gate B: smoke coverage after Priority #1
- Command: `python scripts/06_smoke_by_domain.py --per-domain 2`
- Required outcome:
  - run completes across all five domains
  - no obvious retrieval collapse in any domain
  - evidence remains doc-scoped and more clause/number/date specific, not noisier

Validation Gate C: sample20 end-to-end after Priority #1
- Command: `python scripts/07_run_sample.py --a-board-quality --sample-size 20`
- Required outcome:
  - no clear answer-quality regression versus the accepted Phase 1 sample20 baseline artifacts
  - token usage does not rise enough to erase the retrieval gain
  - report-domain misses are specifically inspected, because they determine whether Priority #2 is still needed

Validation Gate D: if Priority #2 is triggered, report-domain retrieval and focused smoke
- Commands:
  - `python scripts/02_build_index.py`
  - `python scripts/05_compare_rag.py --domains financial_reports`
  - `python scripts/06_smoke_by_domain.py --domains financial_reports --per-domain 3`
  - `python scripts/07_run_sample.py --a-board-quality --qids <report-heavy qids>`
- Required outcome:
  - correct row/block evidence appears higher in top-k for report questions
  - evidence packs become more specific without flooding adjacent irrelevant rows

Validation Gate E: final full A100 after all selected Phase 2 work
- Command: `python scripts/07_run_sample.py --a-board-quality --sample-size 100`
- Required outcome:
  - produces a full A100 artifact set comparable to the Phase 1 A100 record
  - any claim about success must be based on this full run (and official submission later if requested), not on proxy or sample-only results

## Stop/go criteria after the first Phase 2 experiment

GO straight to full A100 with Priority #1 only if all of the following are true:
1. Gate A shows a real proxy retrieval improvement over `field_boosted_rrf` on `ALL` metrics.
2. Gate B shows no domain collapse.
3. Gate C shows stable or improved answer behavior at roughly comparable cost.
4. Financial-report failures do not remain the dominant unresolved error mode.

GO to Priority #2 before full A100 if either of the following is true:
1. Priority #1 helps broadly but financial-report retrieval is still the main visible weakness in Gate C.
2. Proxy/sample evidence shows that decisive report rows remain buried inside coarse whole-table chunks.

STOP and do not proceed to Priority #2 or full A100 yet if any of the following happens:
1. Gate A does not beat `field_boosted_rrf` on the main `ALL` ranking metrics.
2. Gate B or C shows a cross-domain regression that outweighs the retrieval gain.
3. The implementation only wins on noisy proxy metrics while end-to-end evidence quality or answer behavior gets worse.

## Executable Phase 2 graph draft for C2

Assignee mapping used here:
- `afac2026-rag-research`: retrieval implementation and retrieval-focused validation
- `default`: review/orchestration and final run execution

### Card 1
Title: `phase2-p1: implement BM25F-lite field-aware sparse scoring`
Assignee: `afac2026-rag-research`
Parents: none beyond the convergence task that unlocks the graph-creation card
Body:
- Implement Priority #1 exactly as the first Phase 2 retrieval experiment.
- Preserve the Phase 1 planner/solver/runtime contract.
- Touch only the Priority #1 files listed in this note unless a minimal adjacent change is required.
- Deliverable: code + tests for the new BM25F-lite variant and, if validated, the live retrieval path hook.
- Required local verification: targeted unit tests for retrieval variants and any changed retrieval logic.
- On completion, include changed files and exact commands run.

### Card 2
Title: `phase2-p1: validate BM25F-lite with proxy + smoke + sample20`
Assignee: `afac2026-rag-research`
Parents: Card 1
Body:
- Run Validation Gates A, B, and C from this note.
- Produce one concise verdict: `go-a100`, `go-p2`, or `stop`.
- Required artifacts: output directories / reports for compare-rag, smoke, and sample20.
- Required metadata: proxy deltas, token deltas, and a short list of remaining failure modes.
- Do not claim benchmark improvement from proxy retrieval metrics.

### Card 3
Title: `phase2-p1: stop-go review and branch selection`
Assignee: `default`
Parents: Card 2
Body:
- Read the Card 2 artifacts and apply the stop/go criteria from this note.
- If verdict is `go-a100`, create the final full A100 evaluation card directly and complete this card.
- If verdict is `go-p2`, create the Priority #2 implementation + validation cards plus the final A100 card, with correct parent links, then complete this card.
- If verdict is `stop`, block the card with a concrete reason and do not create downstream implementation cards.
- This card is the branch point; do not leave the result as prose only.

### If Card 3 chooses Priority #2, create these downstream cards

Card 4
Title: `phase2-p2: implement T-RAG-lite row-block table retrieval`
Assignee: `afac2026-rag-research`
Parents: Card 3
Body:
- Implement row/block table chunking and the minimum context/index changes listed in this note.
- Preserve reversibility: dual-path or flaggable rollout, not destructive replacement.
- Deliverable: code + tests + rebuild instructions.

Card 5
Title: `phase2-p2: validate row-block retrieval on reports + focused sample`
Assignee: `afac2026-rag-research`
Parents: Card 4
Body:
- Run Validation Gate D from this note.
- Report whether report-domain evidence quality improved enough to justify full A100.
- Include exact qids used for any focused sample runs.

Card 6
Title: `phase2-final: full A100 evaluation after selected Phase 2 retrieval upgrades`
Assignee: `default`
Parents:
- if Card 3 verdict is `go-a100`: parent should be Card 3
- if Card 3 verdict is `go-p2`: parent should be Card 5
Body:
- Run the dedicated final full A100 evaluation only after all selected implementation and validation work is complete.
- Command: `python scripts/07_run_sample.py --a-board-quality --sample-size 100`
- Required outputs: full artifact directory, token report, answer outputs, and a short comparison against the Phase 1 A100 baseline artifacts.
- Do not turn proxy retrieval metrics into benchmark claims.

## Bottom line

The Phase 1 baseline is stable enough to freeze. The best first Phase 2 move is BM25F-lite because it is the cheapest broad retrieval-quality upgrade on the current sparse substrate. T-RAG-lite row-block retrieval should be the immediate fallback only if the first experiment confirms that financial-report table granularity is still the dominant remaining bottleneck after BM25F-lite.