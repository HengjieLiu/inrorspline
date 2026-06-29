# OASIS DirectCP 8-4-2 Full Alpha Sweep

- Generated: `2026-06-26 20:03:28 +0000`
- Remaining-root: `/database/inrorspline/runs/oasis_seed3_first20_directcp842_alpha_sweep_remaining5_v1`
- Reused alpha1000 root: `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_seed3_first20_directcp842_alpha_sweep`
- Remaining summaries: `100`
- Reused alpha1000 summaries: `20`
- Combined summaries: `120`
- Alpha grid: `0,10,100,1000,10000,100000`; alpha `1000` is reused, not rerun.

## Acceptance

- PASS: all expected combined DirectCP 8-4-2 rows and artifacts are present.

## Best Alpha Summary

- first10: best Dice alpha `1e+03` (Dice `0.8054`), best HD95 alpha `1e+04`, best symmetric ASD alpha `1e+03`.
- first20: best Dice alpha `1e+03` (Dice `0.8077`), best HD95 alpha `1e+04`, best symmetric ASD alpha `1e+03`.

## first10

| alpha | n | dice mean | hd95 mean | asd symmetric mean | be native analytic fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10 | 0.7177 | 4.9287 | 0.9131 | 9.2436 | 1.7924 | 1.4755 | 36.8583 | 1.1749 |
| 10 | 10 | 0.7844 | 3.3540 | 0.5755 | 1.6105 | 0.5756 | 1.3015 | 19.4498 | 0.1851 |
| 100 | 10 | 0.8004 | 2.5690 | 0.4844 | 0.4381 | 0.2983 | 1.1001 | 9.5869 | 0.0510 |
| 1e+03 | 10 | 0.8054 | 2.2429 | 0.4507 | 0.1137 | 0.1639 | 0.8826 | 4.3015 | 0.0147 |
| 1e+04 | 10 | 0.7911 | 2.1260 | 0.4690 | 0.0247 | 0.0790 | 0.6501 | 1.1600 | 0.0022 |
| 1e+05 | 10 | 0.7487 | 2.3391 | 0.5907 | 0.0035 | 0.0280 | 0.3557 | 0.0557 | 6.679e-05 |

## first20

| alpha | n | dice mean | hd95 mean | asd symmetric mean | be native analytic fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 20 | 0.7195 | 4.8381 | 0.9053 | 9.6543 | 1.8855 | 1.4876 | 37.0443 | 1.3407 |
| 10 | 20 | 0.7858 | 3.2927 | 0.5778 | 1.6559 | 0.5977 | 1.3209 | 19.7257 | 0.1908 |
| 100 | 20 | 0.8031 | 2.5379 | 0.4867 | 0.4418 | 0.3054 | 1.1217 | 9.6993 | 0.0515 |
| 1e+03 | 20 | 0.8077 | 2.2589 | 0.4531 | 0.1150 | 0.1681 | 0.9038 | 4.4071 | 0.0152 |
| 1e+04 | 20 | 0.7959 | 2.1119 | 0.4580 | 0.0251 | 0.0812 | 0.6674 | 1.2572 | 0.0024 |
| 1e+05 | 20 | 0.7574 | 2.2899 | 0.5668 | 0.0036 | 0.0290 | 0.3731 | 0.1497 | 8.008e-05 |

## Files

- `summary_runs_combined.csv/json`: all 120 DirectCP 8-4-2 summaries.
- `summary_runs_remaining5.csv/json`: the 100 newly launched remaining-alpha summaries.
- `summary_runs_alpha1000_reused.csv/json`: the 20 reused alpha1000 summaries.
- `population_summary_directcp842_first10/first20.csv/json`: first10 and first20 population curves.
- `stage_history_combined.csv`, `stage_summaries_combined.csv`, and `prolongation_metrics_combined.csv`: multistage diagnostics.
