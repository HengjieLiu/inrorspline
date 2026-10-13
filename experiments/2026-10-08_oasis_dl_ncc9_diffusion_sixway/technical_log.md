# Technical log

## Known limitations

- First10 test-case tuning is exploratory; no unbiased benchmark claim.
- Full-resolution loss matches the DL per-direction objective, not its network
  optimizer, bidirectional learning, architecture or learned prior.
- Pyramid mode deliberately changes the loss evaluation scale. Its coordinate
  alignment is corrected rather than copying the historical shortcut.
- Pipeline compute budgets differ; concurrent workloads invalidate clean timing
  comparisons. cuDNN deterministic is not global CUDA bitwise determinism.
- Best-checkpoint selection at every stage changes historical terminal handoff.

## 2026-10-08

- Recorded the approved read-only audit and complete execution specification.
- Implemented an isolated numerical core, worker, dependency scheduler, and
  complete-cohort report generator. Historical runners were not modified by
  this implementation.
- Passed 16 new unit tests and 39 discoverable OASIS unittest tests in total.
  Coverage includes native loss/gradient parity, endpoint-aligned geometry,
  full-grid legacy expansion parity, exact knot insertion, state-indexed best
  checkpoints, CLI/cache identity, canonical exclusions and plot export.
- All six full-volume, one-update-per-stage GPU smokes completed successfully
  at 13:59 UTC. Each passed DVF serialization, canonical metrics, finite-field,
  source/checkpoint/hash and completion-marker checks. Smoke metrics are
  functional checks, not scientific performance results.
- Native CUDA NCC9 and diffusion value and gradient differences were exactly
  zero in all six parity checks. Dense initialization and prefix reconstruction
  differences were exactly zero. Largest knot-insertion field difference was
  1.662e-6 native voxel, consistent with FP32 operation-order rounding.
- Maximum smoke allocated CUDA memory was 1.464 GiB for842, 1.987 GiB for8421,
  and 1.638 GiB for842-to-dense. This excludes other jobs and CUDA context/cache
  overhead; scheduler records fresh device memory at every launch.
- Production started at 14:04 UTC after the smoke gate using GPUs0,1,2,3,
  max-concurrent4 and UID:GID1009:1020. The scheduler revalidated smoke outputs
  and immutable references, then began 140 prefixes and 280 refinements. No
  existing GPU job was stopped or modified.
- At 14:07 UTC, the scheduler and four workers on distinct GPUs were verified
  alive. The first production registrations had passed completion validation;
  no failed attempts were recorded. Full-cohort results remain pending.
- Aggregation remains gated on all 420 production outputs. Numerical source
  hashes are frozen; modifying those sources during the sweep causes a stop
  rather than silently mixing implementations. No final result is claimed yet.
