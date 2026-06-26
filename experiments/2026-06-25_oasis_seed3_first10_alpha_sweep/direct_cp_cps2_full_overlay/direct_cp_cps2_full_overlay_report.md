# OASIS Seed-3 First10 DirectCP cps2 Full Overlay

- Generated: `2026-06-25 22:43:54 +0000`
- Original root: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`
- DirectCP cps2 preliminary root: `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1`
- DirectCP cps2 completion root: `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_completion_a0_10_100000_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay`
- Original completed summaries: `240` / `240`
- DirectCP cps2 summaries: `60` / `60`
- Case ranks: `0,1,2,3,4,5,6,7,8,9`
- DirectCP cps2 alphas: `0,10,100,1000,10000,100000`
- DirectCP cps2 counts by alpha: `0:10, 10:10, 100:10, 1000:10, 10000:10, 100000:10`
- DirectCP cps2 counts by rank: `000:6, 001:6, 002:6, 003:6, 004:6, 005:6, 006:6, 007:6, 008:6, 009:6`
- Original first10 report, figures, and preliminary cps2 overlay are not overwritten.

## Acceptance

- PASS: original first10 and all six DirectCP cps2 alpha rows are present with expected metadata and artifacts.

## Best Alpha By Population Mean

- IDIR: best mean Dice alpha `1`, best mean HD95 alpha `1`, best mean symmetric ASD alpha `1`.
- SINR cps4: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- SINR cps2: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- DirectCP cps4 lr1e-3: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- DirectCP cps2 lr1e-3: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `10000`.

## Files

- `summary_runs_with_direct_cp_cps2_full.csv/json`: original plus full DirectCP cps2 run summaries.
- `population_summary_with_direct_cp_cps2_full.csv/json`: method/alpha population means and spreads.
- `population_label_metrics_with_direct_cp_cps2_full.csv`: per-label population aggregates.
- `case_metrics_all_with_direct_cp_cps2_full.csv`: per-run Dice/HD95/ASD with per-label columns.
- `regularity_metrics_all_with_direct_cp_cps2_full.csv`: per-run full-field and foreground regularity metrics.
- `be_metrics_all_with_direct_cp_cps2_full.csv` and `diffusion_metrics_all_with_direct_cp_cps2_full.csv`: long-form regularity audit rows.
- `population_arc_primary_with_direct_cp_cps2_full.pdf`: focused overlay ARC-style Dice-vs-regularity curves.
- `population_pareto_contact_sheet_with_direct_cp_cps2_full.pdf`: all overlay accuracy-regularity population curves.
- `loss_curves_population_with_direct_cp_cps2_full.pdf`: overlay loss curves.

