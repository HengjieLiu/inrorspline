# Codex Execution Spec: OASIS DirectCP 8-4-2 lr=1e-3 Standardization

## Objective

Replace the final-paper OASIS DirectCP 8-4-2 Pareto/table temporary `lr=1e-2`
exception with a canonical `lr=1e-3` candidate, without overwriting existing
`v001` artifacts.

## Run Scope

- Dataset: OASIS BrainMR seed-3 shuffled first20 ranks `0-19`.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Alpha grid: `0,10,100,1000,10000,100000`.
- LR: `1e-3`.
- Objective: full-image global NCC plus `alpha *` analytic cubic B-spline BE.
- Schedule: cps `8,4,2`; downsample `4,2,1`; stage epochs `400,800,1300`.
- Total epochs: `2500`.
- Seed: `1`.
- Evaluation interval: `250`.
- Docker container: `py310sitreg_v1_test1`.

## Outputs

- Smoke root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_smoke_v1`
- Full root:
  `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1`
- Review root:
  `experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/`
- Candidate artifact roots:
  - `experiments/paper_artifacts/best_result_table/v002_candidate/`
  - `experiments/paper_artifacts/pareto_figure/v002_candidate/`
  - `experiments/paper_artifacts/comparison/v001_vs_v002_candidate/`

## Acceptance

- Smoke run compiles and produces finite metrics.
- Full run completes `120/120` case-alpha jobs.
- Aggregation validates all expected case-alpha combinations with no errors.
- `v002_candidate` best-result table and Pareto validation reports have zero
  `temporary_exception`, `mismatch`, or blocking LR issues.
- Existing `v001` folders are not overwritten.
