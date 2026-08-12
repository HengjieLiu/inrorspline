# Codex Execution Spec: OASIS MR-SINR 8/4/2 Pilot

## Scope

Implement and run the one-case OASIS MR-SINR 8/4/2 pilot from `plan.md`.

## Code Changes

- Add `mr_sinr842_composed` to `scripts/experiments/oasis_alpha_sweep/run_one.py`.
- Add `mr_sinr842_requery` to `scripts/experiments/oasis_alpha_sweep/run_one.py`.
- Preserve the existing OASIS loader, split, image/label direction, full-image
  SINR NCC objective, and fixed-image `image > 0` foreground metric convention.
- Add a pilot launcher and aggregator for reproducible rank-0 comparison.

## Fixed Pilot Command Parameters

```text
case_rank: 0
expected_pair: (0333, 0115)
expected_original_case_index: 158
alpha: 1000
epochs: 2500
stage_full_cps: 8,4,2
stage_downsample: 4,2,1
stage_epochs: 400,800,1300
lr: existing SINR default, 1e-4
spline_omega: existing SINR default, 50.0
```

## Outputs

- Per-run normal `run_one.py` artifacts under
  `<container-data-root>/inrorspline/runs/oasis_mr_sinr842_revision_pilot_v1`.
- Experiment-level review artifacts in this folder:
  - `job_manifest.json`
  - `pilot_comparison_summary.csv/json`
  - `pilot_decision.json`
  - `pilot_report.md`
  - combined case, regularity, BE, diffusion, stage, and diagnostic CSVs.

## Verification

- Compile changed Python scripts.
- Validate rank-0 case resolution before launch.
- Run the launcher manifest dry-run.
- After both pilot runs complete, run the aggregator and inspect
  `pilot_decision.json` before any first10 or first100 scale-up.
