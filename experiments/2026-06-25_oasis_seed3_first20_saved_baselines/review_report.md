# OASIS Seed-3 first20 Saved Baseline Evaluation

- Generated: `2026-06-26 01:05:44 +0000`
- Baseline output root: `/database/inrorspline/runs/oasis_seed3_first10_saved_baselines_v1`
- Reference review root: `/homebase/code_sync/inrorspline/experiments/nonexistent_reference_for_first20_baseline_only`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first20_saved_baselines`
- Saved baseline summaries: `100` / `100`
- Reference summaries loaded from review CSV: `0`
- Case ranks: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- Baselines are saved dense displacement fields; training-time fields are not applicable.
- Baseline regularity uses dense finite-difference metrics. Native analytic B-spline/IDIR columns are intentionally unavailable for these rows.

## Acceptance

- PASS: all requested saved baseline rows are present with expected cases, metadata, and artifacts.

## Saved Baseline Population Means

- BS1 VoxelMorph: mean Dice `0.7724`, mean HD95 `2.5298`, mean symmetric ASD `0.5493`.
- BS2 TransMorph: mean Dice `0.7954`, mean HD95 `2.0932`, mean symmetric ASD `0.4483`.
- BS3 VFA: mean Dice `0.8223`, mean HD95 `1.8050`, mean symmetric ASD `0.3638`.
- BS4 SITReg: mean Dice `0.8245`, mean HD95 `1.7505`, mean symmetric ASD `0.3491`.
- Greedy: mean Dice `0.8067`, mean HD95 `1.9405`, mean symmetric ASD `0.4101`.

## Files

- `summary_runs_saved_baselines.csv/json`: saved baseline run summaries only.
- `summary_runs_with_saved_baselines.csv/json`: reference summaries, if provided, plus saved baseline rows.
- `population_summary_with_saved_baselines.csv/json`: method population aggregates.
- `population_label_metrics_with_saved_baselines.csv`: per-label population aggregates.
- `population_arc_fd_with_saved_baselines.pdf`: primary Dice-vs-FD/Jacobian overlay.
- `population_pareto_contact_sheet_with_saved_baselines.pdf`: all accuracy-vs-FD/Jacobian overlay panels.

