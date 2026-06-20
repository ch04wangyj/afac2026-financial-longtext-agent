# Phase 1 baseline report: paper-faithful LogicRAG, retained extensions, and Phase 2 handoff

## Purpose

This report is the formal Phase 1 handoff baseline for AFAC2026. It records what the repo now treats as the paper-faithful LogicRAG core, which project-specific extensions were intentionally retained, which behaviors were removed from the default baseline or deferred, what validation/score/token artifacts back the baseline, and which retrieval limitations still remain after Phase 1 alignment.

Primary grounding sources:
- `theory/references/notes/2026-06-19_logicrag-2508.06105v2-grounded-spec.md:7-20,53-70,91-157,179-199`
- `theory/references/notes/2026-06-19_logicrag-current-vs-paper-gap-matrix.md:7-15,20-29,31-59`
- `configs/logicrag_runtime.yaml:1-77`
- `tests/test_logicrag_plan.py:35-85`
- `tests/test_logicrag_retrieval.py:61-138`
- `tests/test_logicrag_solver.py:55-173,203-260`
- `tests/test_logicrag_config.py:13-121`
- `tests/test_reasoning_prompts.py:49-74`
- `VERSION_SCORE_LOG.md:74-80`
- `outputs/phase1_validation_sample20_2026-06-19_155545/run_report.md:1-56`
- `outputs/phase1_validation_a100_2026-06-19_160310/run_report.md:1-140`

## Executive summary

Phase 1 did achieve a narrower, more explicit LogicRAG baseline than the pre-alignment stack. The repo now encodes a paper-faithful core contract around:
1. LLM-generated subproblem decomposition,
2. dependency DAG validation/topological ranking,
3. same-rank merged retrieval scheduling,
4. rolling rank-memory summarization,
5. explicit runtime defaults for dynamic augmentation / forced forward progress / paper-core mode.

However, this is still not a literal paper reproduction. The strongest remaining mismatch is retrieval: runtime config now names the paper contract as embedding-backed, but the live AFAC implementation still runs on sparse BM25/RRF plus local retrieval heuristics. Dynamic augmentation is represented in schema/config/tests, but Phase 1 did not finish the full unresolved-node execution path in the production solver. Therefore the correct claim is:
- Phase 1 delivered a paper-faithful execution boundary and baseline semantics.
- Phase 1 did not yet close the retrieval-backend and dynamic-augmentation gaps needed for a fully paper-native LogicRAG implementation.

## 1. What counts as the Phase 1 paper-faithful core

The paper-grounded contract note defines the non-negotiable LogicRAG behaviors as LLM decomposition, DAG dependencies, topological/rank-wise execution, same-rank unified retrieval, rolling summarized memory, dynamic graph growth when retrieval is insufficient, sampling without replacement, and moderate top-k operation (`...grounded-spec.md:9-20,57-70,93-157,189-199`).

Phase 1 baseline keeps the following core behaviors in-repo:

### 1.1 LLM decomposition + dependency DAG as first-class runtime objects
- `tests/test_logicrag_plan.py:14-33` verifies subproblems parse into a `LogicPlan` and preserve topological levels.
- `tests/test_logicrag_plan.py:87-103` verifies the planner prompt explicitly tells the model that `depends_on` expresses logical prerequisites and that same-rank subproblems will later merge into one retrieval.
- `tests/test_logicrag_plan.py:35-64` verifies sanitization metadata records the planner contract and distinguishes paper-faithful core from applied local extensions.

Operationally, Phase 1 baseline means LogicRAG is no longer described as a generic retrieval variant; it is a plan-bearing execution mode with explicit node/rank structure.

### 1.2 Same-rank merge scheduling and memory-conditioned later retrieval
- `tests/test_logicrag_retrieval.py:110-138` verifies rank-wise retrieval runs in multiple stages, rank 1 queries inherit an upstream memory anchor, and same-rank query groups are preserved.
- `tests/test_logicrag_solver.py:100-137` verifies later-rank queries incorporate prior memory rather than re-asking the same raw question.
- `tests/test_logicrag_solver.py:203-227` verifies rank-wise query building does not mutate its own query list while iterating and only injects the memory anchor once.

This is the main paper-faithful execution improvement over the older flatter retrieval-first variants.

### 1.3 Summary-first carry-forward instead of full raw-evidence accumulation
- `tests/test_logicrag_solver.py:55-98` verifies solver metadata returns compact `rank_memories` with only `rank`, `summary`, and `evidence_doc_ids`.
- `tests/test_logicrag_solver.py:139-173` verifies final compose uses the last-rank evidence instead of accumulated raw context from all previous ranks.
- `tests/test_reasoning_prompts.py:64-74` verifies the prompt contract explicitly says rank memory is preferred over raw evidence pile-up and that missing evidence should lower confidence rather than trigger token-heavy compensation.

This is the cleanest concrete Phase 1 alignment to the paper's context-pruning story.

### 1.4 Paper-core runtime defaults are now explicit, not implicit
- `configs/logicrag_runtime.yaml:49-78` sets:
  - `execution_mode: paper_faithful_core`
  - `retrieval_backend: paper_contract_embedding`
  - `dynamic_dag_augmentation: true`
  - `append_unresolved_after_current_rank: true`
  - `sampling_without_replacement: true`
  - `rank_top_k: 5`
- `tests/test_logicrag_config.py:53-61` verifies these defaults.
- `tests/test_logicrag_config.py:25-50` verifies the step-specific budget hierarchy, including rank-top-k=5 and planner/final-compose/rank-summary thinking budgets.

This matters because the repo now names the intended baseline directly instead of burying it under generic hybrid retrieval defaults.

### 1.5 Prompt contract now forbids "token compensation" for weak retrieval
- `tests/test_reasoning_prompts.py:49-55` verifies answer prompts tell the model not to compensate for retrieval misses with broad speculative reasoning.
- `tests/test_reasoning_prompts.py:57-62` verifies planner prompts require concrete retrieval targets instead of background/expository subproblems.

That is an important Phase 1 hardening step: the baseline is now allowed to be uncertain, rather than pretending missing evidence can be solved by generating more prose.

## 2. Retained project-specific extensions in the baseline

Phase 1 was not a pure delete-everything exercise. Several AFAC-specific extensions remain intentionally retained because they stabilize the competition setting or support sparse-first execution.

### 2.1 Planner sanitization and safety rails remain
`tests/test_logicrag_plan.py:35-64` shows the retained extensions explicitly recorded in metadata:
- `drop_missing_dependencies`
- `break_cycles`
- `drop_duplicate_subproblems`
- `drop_empty_subproblems`
- `trim_excess_ranks`

These are not paper-native behaviors, but they are pragmatic guards against malformed LLM plans.

### 2.2 Sparse retrieval target engineering remains available
`agent/retrieve/targets.py:12-222` retains AFAC-specific sparse retrieval target construction:
- must/should term extraction,
- numbers/dates/entities/option-term extraction,
- query variant generation,
- merged same-rank retrieval targets.

The gap matrix classifies this as an additive extension rather than a paper requirement (`...gap-matrix.md:23,46-50`). It survives because the live retrieval substrate is still sparse-first.

### 2.3 Deterministic evidence-pack expansion remains
`agent/retrieve/context.py:9-206` still expands anchor chunks by same clause, same section, same page for tables/figures, and local neighbors; it also enforces doc-balanced pack selection when multiple `doc_ids` exist (`agent/retrieve/context.py:21-63,67-115,142-206`).

This is useful for AFAC evidence assembly, but it is not the paper's context-pruning definition (`...gap-matrix.md:25,46-50`). It should continue to be described as a local evidence-pack heuristic.

### 2.4 A-board multi-option fallback remains enabled
`configs/logicrag_runtime.yaml:67-78` keeps:
- `multi_logicrag_enabled: true`
- `multi_logicrag_retry_enabled: true`
- `force_doc_coverage_for_a_board: true`
- `use_doc_ids_as_hint_only: false`

`agent/reasoning/multi_logicrag.py:58-97` keeps the AFAC policy that every uncertain option may receive one wider retrieval pass, with option-specific retry queries built from entities/numbers/dates.

This is a deliberate retained competition extension. It is useful for AFAC multi-select behavior, but it is beyond the paper-faithful LogicRAG core and should stay labeled that way.

### 2.5 Thinking-budget hierarchy remains an AFAC runtime optimization
The YAML thinking profiles and the corresponding config tests (`configs/logicrag_runtime.yaml:7-47`, `tests/test_logicrag_config.py:13-50,63-103`) retain step-specific token budgeting. This is not part of the paper algorithm, but it is a practical runtime-control layer that Phase 1 chose to keep.

## 3. Removed from the default baseline or explicitly deferred

Phase 1 did not remove every extension from the codebase, but it did move several behaviors out of the claimed default baseline.

### 3.1 Large top-k default removed from the baseline
The earlier runtime had `rank_top_k: 12` per the audit note (`...gap-matrix.md:28`). Phase 1 baseline now sets `rank_top_k: 5` in `configs/logicrag_runtime.yaml:56-59`, matching the paper's moderate top-k claim (`...grounded-spec.md:141-157`).

### 3.2 "Generic hybrid mode" is no longer the intended LogicRAG story
The audit note flagged `agent/config.py` defaulting to generic hybrid retrieval and optional LogicRAG enablement as a mismatch (`...gap-matrix.md:29`). Phase 1 baseline answers this by naming a dedicated `paper_faithful_core` LogicRAG mode in runtime config, even though the broader repo still supports other retrieval paths.

### 3.3 Query-expansion/rerank/context helpers are no longer part of the paper-faithful claim
The gap matrix explicitly places the following in "gate or postpone / adaptation-only" status (`...gap-matrix.md:24-26,46-51`):
- lexical feedback query expansion,
- deterministic context-pack heuristics,
- sparse heuristic reranking,
- A-board multi-option retry/coverage logic.

Phase 1 baseline therefore treats these as retained AFAC modules, not as evidence that the implementation fully matches the paper.

### 3.4 Optional A-board adjuncts are disabled in the baseline config
`configs/logicrag_runtime.yaml:67-78` keeps several adjunct controls off by default:
- `option_matrix_enabled: false`
- `coverage_gate_enabled: false`
- `financial_calculator_enabled: false`

This matters for handoff: the validated Phase 1 baseline is the narrower path defined by the current YAML, not a union of every optional subsystem in the repo.

## 4. Validation, score, token, and artifact record

### 4.1 Official score record
`VERSION_SCORE_LOG.md:74-80` records the accepted Phase 1 version as:
- Version: V7
- Date: 2026-06-19
- Official score: `39.6108`
- Delta vs previous round: `-0.3856`
- Meaning: this is the validated paper-faithful-baseline submission, not merely an offline proxy result.

### 4.2 Validation artifacts kept on disk
Phase 1 validation artifacts are present in:
- `outputs/phase1_validation_sample20_2026-06-19_155545/`
- `outputs/phase1_validation_a100_2026-06-19_160310/`

Sample20 directory contents include:
- `answer_results.jsonl`
- `answer.csv`
- `evidence.json`
- `token_usage.json`
- `run_report.md`
- `run_report.json`
- `results.csv`
- `sample_manifest.json`
- baseline comparison / answer-delta reports

A100 directory contents include:
- `answer_results.jsonl`
- `answer.csv`
- `evidence.json`
- `token_usage.json`
- `run_report.md`
- `run_report.json`
- `results.csv`
- `run.log`
- V5/V6 delta reports

### 4.3 Sample20 token and issue profile
From `outputs/phase1_validation_sample20_2026-06-19_155545/run_report.md:3-56` and `.../token_usage.json:1-4`:
- Total tokens: `284,971`
- Prompt tokens: `201,756`
- Completion tokens: `83,215`
- Domain totals:
  - financial_contracts: `41,933`
  - financial_reports: `53,988`
  - insurance: `76,347`
  - regulatory: `49,994`
  - research: `62,709`
- Logged low-confidence items: `ins_a_001`, `ins_a_004`, `reg_a_018`

The sample20 run is important because it shows the baseline is not hiding uncertainty; some questions remain flagged low-confidence even after alignment.

### 4.4 Full A100 token and issue profile
From `outputs/phase1_validation_a100_2026-06-19_160310/run_report.md:3-140` and `.../token_usage.json:1-4`:
- Total tokens: `1,662,572`
- Prompt tokens: `1,378,610`
- Completion tokens: `283,962`
- Domain totals:
  - financial_contracts: `292,615`
  - financial_reports: `293,689`
  - insurance: `410,299`
  - regulatory: `275,957`
  - research: `390,012`
- Logged low-confidence items: `fc_a_010`, `fc_a_018`, `fin_a_011`, `fin_a_015`, `reg_a_008`, `reg_a_013`, `reg_a_018`

Format-level totals in the same report show the strongest token load is still multi-select execution (`1,220,765` total tokens for multi questions), which is relevant for Phase 2 cost-control planning.

### 4.5 Regression / test evidence for the baseline contract
I re-ran the focused Phase 1 LogicRAG test suite in this task:
- Command: `python -m pytest tests/test_logicrag_plan.py tests/test_logicrag_retrieval.py tests/test_logicrag_solver.py tests/test_logicrag_config.py tests/test_reasoning_prompts.py`
- Result: `35 passed` in `0.52s`

This provides live verification that the current repo still satisfies the core Phase 1 planner/retrieval/solver/config/prompt contracts described above.

## 5. Retrieval limitations still outstanding after Phase 1 alignment

These are the main reasons Phase 1 should be treated as a baseline handoff, not the end-state.

### 5.1 Retrieval backend mismatch is still the largest unresolved paper gap
The paper requires an embedding retriever `R(q; θ, K)` (`...grounded-spec.md:19,176-177,199`). The gap matrix states the live retriever is still BM25 + optional doc-level search + sparse query fusion (`...gap-matrix.md:22,39-45`).

So even though runtime config now names `retrieval_backend: paper_contract_embedding`, the production implementation remains sparse-first. This is the clearest Phase 2 convergence target.

### 5.2 Dynamic DAG augmentation is specified, but not fully closed in live solver semantics
The paper requires runtime augmentation when retrieval is insufficient (`...grounded-spec.md:53-70`).
- `tests/test_logicrag_plan.py:66-85` proves plan objects can append unresolved subproblems after the current rank sequence.
- `configs/logicrag_runtime.yaml:51-55` turns the feature on at the config-contract layer.

But the gap matrix still calls out missing dynamic augmentation in the actual execution path (`...gap-matrix.md:20-21,39-45`). In other words: the repo now has the contract and some machinery, but Phase 2 still has to make it a production behavior instead of a partially represented capability.

### 5.3 Solver still is not a full node-resolution-forward implementation
The gap matrix states the current solver still does not truly resolve each node/subproblem into its own answer before moving on, and still relies on a final compose stage (`...gap-matrix.md:21,42-44`).

Phase 1 improved this materially by switching the compose step to rank-memory-first and by preventing raw-context accumulation, but it did not yet fully become the paper's per-node answer-resolution pipeline.

### 5.4 AFAC retrieval remains adaptation-heavy in the sparse stack
The retained sparse stack still depends on local adaptation layers:
- lexical target engineering (`agent/retrieve/targets.py`),
- evidence-pack heuristics (`agent/retrieve/context.py`),
- uncertain-option retry (`agent/reasoning/multi_logicrag.py`).

These are helpful for AFAC, but they mean current performance is not solely attributable to the paper-core LogicRAG contract.

### 5.5 Financial-report retrieval is still structurally underpowered
The same-day Phase 2 notes repeatedly identify table granularity and field-aware retrieval as the next likely upside areas:
- `theory/references/notes/2026-06-19_afac-shortlist-experiment-matrix.md:9-23,88-113`
- `theory/references/notes/2026-06-19_retrieval-improvement-paper-inventory.md:11-28,220-237`
- `theory/references/notes/2026-06-19_afac-fit-analysis-sparse-first.md:15-38,113-141,145-179,233-239`

This matches the A100 report, where financial-report questions still contribute substantial token cost and low-confidence cases (`run_report.md:14,31-32,61-80`). Phase 1 aligned execution semantics more than it solved the financial-report retrieval substrate.

## 6. Recommended interpretation for Phase 2 convergence

Phase 2 should treat Phase 1 as a frozen baseline with this exact wording:
- The repo now has a validated paper-faithful LogicRAG execution boundary.
- The validated score attached to that boundary is `39.6108` (V7).
- The boundary still runs on a sparse AFAC retrieval substrate and still carries AFAC-specific adaptation modules.
- Therefore Phase 2 should converge by upgrading the retrieval substrate and unresolved-node execution path, not by reopening the entire planner/solver contract.

Concretely, the best next-step interpretation is:
1. Preserve the Phase 1 planner/rank-memory/prompt/runtime contract as the baseline.
2. Treat embedding-vs-sparse retrieval substrate, dynamic augmentation completion, field-aware sparse scoring, and table-granular retrieval as the main open fronts.
3. Measure all future gains against the recorded Phase 1 artifacts instead of against older pre-alignment variants.

## Bottom line

Phase 1 successfully narrowed AFAC LogicRAG into a defensible baseline: explicit paper-core mode, moderate top-k, rank-wise memory propagation, summary-first final compose, and validated sample20/A100 artifacts behind official score V7 = `39.6108`.

But Phase 1 did not make the system fully paper-native. Retrieval is still sparse-first, dynamic augmentation is only partially closed in production semantics, and several AFAC-specific retrieval/control extensions remain intentionally active. That is the correct handoff state for Phase 2: baseline semantics are now stable enough to stop debating the contract and start attacking retrieval reality.