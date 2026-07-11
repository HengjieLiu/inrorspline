# OASIS Seed-3 First10 Saved Baseline Evaluation

- Generated: `2026-06-26 00:53:09 +0000`
- Baseline output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`
- Original first10 review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_saved_baselines`
- Saved baseline summaries: `50` / `50`
- Original first10 summaries loaded from review CSV: `240`
- Case ranks: `0,1,2,3,4,5,6,7,8,9`
- Baselines are saved dense displacement fields; training-time fields are not applicable.
- Baseline regularity uses dense finite-difference metrics. Native analytic B-spline/IDIR columns are intentionally unavailable for these rows.

## Acceptance

- PASS: all requested saved baseline rows are present with expected cases, metadata, and artifacts.

## Saved Baseline Population Means

- BS1 VoxelMorph: mean Dice `0.7683`, mean HD95 `2.4477`, mean symmetric ASD `0.5497`.
- BS2 TransMorph: mean Dice `0.7913`, mean HD95 `2.0084`, mean symmetric ASD `0.4467`.
- BS3 VFA: mean Dice `0.8198`, mean HD95 `1.7743`, mean symmetric ASD `0.3669`.
- BS4 SITReg: mean Dice `0.8215`, mean HD95 `1.7159`, mean symmetric ASD `0.3509`.
- Greedy: mean Dice `0.8037`, mean HD95 `1.8726`, mean symmetric ASD `0.4096`.

## Files

- `summary_runs_saved_baselines.csv/json`: saved baseline run summaries only.
- `summary_runs_with_saved_baselines.csv/json`: original first10 summaries plus saved baseline rows.
- `population_summary_with_saved_baselines.csv/json`: method population aggregates.
- `population_label_metrics_with_saved_baselines.csv`: per-label population aggregates.
- `population_arc_fd_with_saved_baselines.pdf`: primary Dice-vs-FD/Jacobian overlay.
- `population_pareto_contact_sheet_with_saved_baselines.pdf`: all accuracy-vs-FD/Jacobian overlay panels.

