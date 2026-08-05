# Codex Execution Spec: OASIS DirectCP 8-4-2 Runtime Sweep

## Objective

Find faster DirectCP `8-4-2` schedules for OASIS BrainMR while preserving most
of the current best Dice/HD95 performance. This run is intended to support the
reviewer-facing runtime argument by separating optimization time from total
pipeline time.

## Run Scope

- Dataset: OASIS BrainMR seed-3 shuffled first10 ranks `0-9`.
- Smoke ranks: `0,4,8`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Alpha: `1000`, reusing the current best regularity setting.
- Stage cps: `8,4,2`.
- Stage downsample: `4,2,1`.
- Seed: `1`.
- Evaluation interval: `250`.
- Docker container: `py310sitreg_v1_test1`.
- Docker image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Git commit at planning/implementation start: `b9d6c52` on branch `gpu8`,
  with this runtime implementation intentionally added before launch.

## Candidate Matrix

- Fixed schedules: `250,500,800`, `150,300,500`, and `100,200,300`.
- Learning rates: `3e-3`, `1e-2`, `3e-2`, and selected `1e-1` stress tests.
- Decayed stage LR candidates: `3e-2,1e-2,3e-3`.
- Adaptive candidates use max epochs `400,800,1300` with per-stage minimums,
  patience windows, and plateau thresholds recorded in `candidate_manifest.csv`.

## Outputs

- External run root:
  `<external-run-root>/runs/oasis_directcp842_runtime_sweep_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/`.
- Smoke review subroot:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/smoke/`.
- First10 review subroot:
  `experiments/2026-08-04_oasis_directcp842_runtime_sweep/first10/`.

## Acceptance

- Docker/GPU preflight passes with visible idle GPUs.
- Rank-0 validation resolves the seed-3 shuffled case correctly.
- Smoke completes all selected candidate/rank jobs and aggregates without
  missing, duplicate, nonfinite, or protocol-invalid rows.
- Candidates are promoted only if smoke Dice is no worse than matched baseline
  by `0.05` and smoke HD95 is no worse by `0.5 mm`.
- First10 candidates are ranked by runtime and accuracy using:
  DirectCP cps2, SINR cps2, and the current DirectCP `8-4-2` baseline.
