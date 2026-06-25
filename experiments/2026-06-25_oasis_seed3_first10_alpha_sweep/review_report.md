# OASIS Seed-3 First10 Alpha Sweep Review

- Generated: `2026-06-25 18:09:01 +0000`
- Large output root: `/database/inrorspline/runs/oasis_seed3_first10_alpha_sweep_be_2500e_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep`
- Expected runs: `240`
- Completed summaries: `240`
- Case ranks: `0,1,2,3,4,5,6,7,8,9`
- Split: seed-3 shuffled `pair_test_200`; selected ranks are from the first10-after-shuffle protocol.
- Population ARC convention: aggregate each method/alpha across cases first, then draw the Pareto/front curves.
- Native analytic regularity: IDIR uses coordinate-autograd BE/diffusion; spline methods use B-spline analytic BE/diffusion.
- Contact sheets plot native analytic BE/diffusion; spline-only analytic columns remain in CSV/JSON for audit.

## Acceptance

- PASS: all expected runs are present with expected cases, alphas, metadata, and artifacts.

## Best Alpha By Population Mean

- IDIR: best mean Dice alpha `1`, best mean HD95 alpha `1`, best mean symmetric ASD alpha `1`.
- SINR cps4: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- SINR cps2: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.
- DirectCP cps4 lr1e-3: best mean Dice alpha `1000`, best mean HD95 alpha `10000`, best mean symmetric ASD alpha `1000`.

## Files

- `summary_runs.csv/json`: one row per run, including per-label metrics and regularity columns.
- `population_summary.csv/json`: one row per method/alpha, with across-case mean/std/median/min/max.
- `population_label_metrics.csv`: per-label population aggregates.
- `case_metrics_all.csv`: per-run Dice/HD95/ASD with per-label columns.
- `regularity_metrics_all.csv`: per-run full-field and foreground regularity metrics.
- `be_metrics_all.csv` and `diffusion_metrics_all.csv`: long-form BE/diffusion audit rows.
- `population_arc_primary.pdf`: focused ARC-style Dice-vs-regularity curves.
- `population_pareto_contact_sheet.pdf`: all accuracy-regularity population curves.
- `loss_curves_population.pdf`: across-case mean loss curves.
