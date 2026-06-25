# OASIS Seed-3 First10 DirectCP cps2 Preliminary Overlay

- Generated: `2026-06-25 18:58:11 +0000`
- Original root: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`
- Add-on root: `/database/inrorspline/runs/oasis_seed3_first10_direct_cp_cps2_prelim_a100_1000_10000_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_prelim_overlay`
- Original completed summaries: `240` / `240`
- DirectCP cps2 preliminary summaries: `30` / `30`
- Case ranks: `0,1,2,3,4,5,6,7,8,9`
- DirectCP cps2 alphas: `100,1000,10000`
- This is a preliminary 3-alpha DirectCP cps2 overlay, not a complete six-alpha DirectCP cps2 sweep.
- Original first10 report and figures are not overwritten.

## Acceptance

- PASS: original first10 and DirectCP cps2 add-on rows are present with expected metadata and artifacts.

## Best Alpha By Population Mean

- IDIR: best mean Dice alpha `1`, best mean HD95 alpha `1`, best mean symmetric ASD alpha `1`.
- SINR cps4: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- SINR cps2: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- DirectCP cps4 lr1e-3: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- DirectCP cps2 lr1e-3: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `10000`.

## Files

- `summary_runs_with_direct_cp_cps2.csv/json`: original plus add-on run summaries.
- `population_summary_with_direct_cp_cps2.csv/json`: method/alpha population means and spreads.
- `population_label_metrics_with_direct_cp_cps2.csv`: per-label population aggregates.
- `case_metrics_all_with_direct_cp_cps2.csv`: per-run Dice/HD95/ASD with per-label columns.
- `regularity_metrics_all_with_direct_cp_cps2.csv`: per-run full-field and foreground regularity metrics.
- `be_metrics_all_with_direct_cp_cps2.csv` and `diffusion_metrics_all_with_direct_cp_cps2.csv`: long-form regularity audit rows.
- `population_arc_primary_with_direct_cp_cps2.pdf`: focused overlay ARC-style Dice-vs-regularity curves.
- `population_pareto_contact_sheet_with_direct_cp_cps2.pdf`: all overlay accuracy-regularity population curves.
- `loss_curves_population_with_direct_cp_cps2.pdf`: overlay loss curves.
