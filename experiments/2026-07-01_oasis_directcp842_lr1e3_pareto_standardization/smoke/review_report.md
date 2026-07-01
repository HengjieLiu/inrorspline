# OASIS DirectCP 8-4-2 Canonical lr=1e-3 Alpha Sweep

- Generated: `2026-07-01 12:13:55 +0000`
- Out root: `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_smoke_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization/smoke`
- Summaries: `1`
- Alpha grid: `0`
- LR: `0.001`
- Variant: `oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_smoke_v1`

## Acceptance

- PASS: all expected DirectCP 8-4-2 lr=1e-3 rows and artifacts are present.

## Best Alpha Summary

- requested: best Dice alpha `0` (Dice `0.5332`), best HD95 alpha `0`, best symmetric ASD alpha `0`.

## requested

| alpha | n | dice mean | hd95 mean | asd symmetric mean | be native analytic fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.5332 | 4.5466 | 1.4993 | 0.0014 | 8.371e-04 | 0.0590 | 0.0000 | 0.0000 |

## Files

- `summary_runs_combined.csv/json`: all DirectCP 8-4-2 lr=1e-3 summaries.
- `population_summary_directcp842_first10/first20.csv/json`: first10 and first20 population curves.
- `stage_history_combined.csv`, `stage_summaries_combined.csv`, and `prolongation_metrics_combined.csv`: multistage diagnostics.
