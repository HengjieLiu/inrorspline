---
doc_type: codex_execution_spec
status: completed
experiment_id: fit_modes_population_n10_v1
---

# Codex Execution Specification

## Objective

Implement, validate, launch, monitor, and aggregate the locked OASIS10/DIR-LAB10
two-panel fit-modes matrix. Do not create a population figure before all 160
records pass aggregation.

## Implementation scope

- Reuse the audited one-job OASIS and DIR-LAB runners; do not duplicate their
  optimizers in the population launcher.
- Generate a deterministic 20-case cohort and 160-job manifest.
- Support `--dry-run`, `--smoke`, `--resume`, explicit output/review roots,
  automatic GPU `0–3` discovery, one process per GPU, and dynamic work stealing.
- Preserve every existing run directory. Incompatible work must use a new
  versioned output root.
- Reuse only the 24 exact-compatible records in the current source-backed
  selected figure and record their provenance.
- Persist progress/ETA in Markdown and JSON on every scheduler poll.
- Aggregate target power, rRMSE, diagnostic NCC, runtime, peak memory, and
  provenance for every record.

## Required validation order

1. Static compile and CLI-help checks.
2. Full dry-run: exactly 20 cases, 160 unique records, 24 exact reuses, 136
   pending jobs, 10 SITReg targets, and 10 pTVReg targets.
3. Deep target loading/orientation validation for all twenty cases.
4. Isolated 16-job smoke on OASIS rank 0 and DIR-LAB case02.
5. Resume test against completed smoke output without mutation.
6. Full four-GPU launch and monitoring until every record is terminal.
7. Aggregation; require all 160 records to be successful or exact reuse, finite
   endpoints, and all 16 dataset/panel/method population groups to have N=10.

## Failure behavior

- Stop on a cohort-count, target, reuse-compatibility, or output-schema error.
- Keep successful jobs and logs after any optimization failure.
- Report a failure immediately with its job ID, GPU, log, and return code.
- Resume only after the cause is inspected; do not overwrite the failed run.
- An unresolved failed record prevents acceptance and figure generation.

## Deliverables

The experiment directory contains protocol, technical log, compact manifests,
monitor state, optimization index, population statistics, and acceptance
reports. After acceptance, it also contains 20 source-backed case diagnostic
figures, a five-by-two-per-dataset Markdown comparison, a curve audit table,
and a shared-axis manifest. Large logs, models, checkpoints, and dense fields
remain external.
