# Execution log

## 2026-10-06: protocol freeze

- User approved the 24-fit VFA/SITReg by cps by learning-rate matrix.
- Reviewed the repository experiment standards, OASIS seed-3 protocol,
  deterministic CUDA decision, Docker workflow, registry, and existing saved
  displacement loader.
- Added a standalone saved-field B-spline fitting runner and four-GPU
  launcher/validator/aggregator. Existing saved fields and earlier fitting
  scripts remain read-only.
- Preflight, smoke, production, and acceptance results are pending.

## 2026-10-06: preflight and execution

- Ten dependency-free unit contracts passed, including parsing, unique names,
  exact voxel scaling, full-volume objective behavior, all-CPS synthetic
  representation recovery, exact zero-target stability, stopping, LR
  selection, and nested-space tolerance.
- VFA and SITReg target audits passed with raw `[3,160,224,192]` and normalized
  `[160,224,192,3]` shapes, `zyx` voxel components, no reorientation, correct
  case/direction, and recorded source hashes.
- Eight one-step target/CPS smokes passed dense serialization, checkpoint,
  field-shape, finite-value, objective, and canonical registration-metric
  validation.
- The dry run contained exactly 24 unique commands and scheduled cps1 first.
- Production completed 24/24 candidates across GPUs 0-3 as UID:GID
  `1009:1020`. Every candidate completed on attempt 0; no OOM or other retry
  occurred and existing GPU processes were not changed.
- Strict aggregation selected eight fits and passed source consistency,
  canonical absent-label handling, finite dense-field, full-volume/no-reg,
  control-shape, curve-generation, and 0.1%-tolerance nested-space checks.
  No recovery continuation was required.
- Accepted external report:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1/report.md`.
