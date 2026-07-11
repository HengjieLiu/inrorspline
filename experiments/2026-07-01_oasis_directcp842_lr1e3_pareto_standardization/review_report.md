# OASIS DirectCP 8-4-2 Canonical lr=1e-3 Alpha Sweep

- Generated: `2026-07-01 14:32:41 +0000`
- Out root: `/database/inrorspline/runs/oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-07-01_oasis_directcp842_lr1e3_pareto_standardization`
- Summaries: `120`
- Alpha grid: `0,10,100,1000,10000,100000`
- LR: `0.001`
- Variant: `oasis_seed3_first20_directcp842_lr1e3_alpha_sweep_v1`

## Acceptance

- PASS: all expected DirectCP 8-4-2 lr=1e-3 rows and artifacts are present.

## Best Alpha Summary

- first10: best Dice alpha `1e+03` (Dice `0.8070`), best HD95 alpha `1e+04`, best symmetric ASD alpha `1e+03`.
- first20: best Dice alpha `1e+03` (Dice `0.8090`), best HD95 alpha `1e+04`, best symmetric ASD alpha `1e+03`.

## first10

| alpha | n | dice mean | hd95 mean | asd symmetric mean | be native analytic fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 0.7549 | 3.5749 | 0.6567 | 4.9529 | 1.0560 | 1.4066 | 30.0778 | 0.5000 |
| 10 | 10 | 0.7851 | 2.9716 | 0.5422 | 1.4274 | 0.4943 | 1.2479 | 16.6832 | 0.1433 |
| 100 | 10 | 0.8024 | 2.4902 | 0.4706 | 0.4272 | 0.2727 | 1.0581 | 8.6650 | 0.0465 |
| 1e+03 | 10 | 0.8070 | 2.2020 | 0.4412 | 0.1141 | 0.1547 | 0.8571 | 4.0573 | 0.0144 |
| 1e+04 | 10 | 0.7917 | 2.1149 | 0.4655 | 0.0242 | 0.0770 | 0.6363 | 1.0958 | 0.0022 |
| 1e+05 | 10 | 0.7491 | 2.3403 | 0.5890 | 0.0032 | 0.0278 | 0.3535 | 0.0561 | 6.735e-05 |

## first20

| alpha | n | dice mean | hd95 mean | asd symmetric mean | be native analytic fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 0.7578 | 3.5321 | 0.6522 | 5.1164 | 1.0957 | 1.4238 | 30.3146 | 0.5270 |
| 10 | 20 | 0.7890 | 2.8997 | 0.5358 | 1.4679 | 0.5131 | 1.2709 | 16.9971 | 0.1492 |
| 100 | 20 | 0.8048 | 2.4371 | 0.4661 | 0.4308 | 0.2796 | 1.0834 | 8.7645 | 0.0469 |
| 1e+03 | 20 | 0.8090 | 2.1915 | 0.4393 | 0.1149 | 0.1587 | 0.8822 | 4.1521 | 0.0146 |
| 1e+04 | 20 | 0.7967 | 2.0904 | 0.4541 | 0.0247 | 0.0792 | 0.6571 | 1.1891 | 0.0023 |
| 1e+05 | 20 | 0.7575 | 2.2879 | 0.5654 | 0.0033 | 0.0289 | 0.3728 | 0.1482 | 7.675e-05 |

## Files

- `summary_runs_combined.csv/json`: all DirectCP 8-4-2 lr=1e-3 summaries.
- `population_summary_directcp842_first10/first20.csv/json`: first10 and first20 population curves.
- `stage_summaries_combined.csv` and `prolongation_metrics_combined.csv`: compact multistage diagnostics.
- `history_combined.csv` and `stage_history_combined.csv` are opt-in high-volume diagnostics; use `--include-history-csvs` when needed.
