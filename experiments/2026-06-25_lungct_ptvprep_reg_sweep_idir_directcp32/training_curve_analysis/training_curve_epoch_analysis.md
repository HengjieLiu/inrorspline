# LungCT Regularization Sweep Training-Curve Analysis

This analysis combines the 40 nonbaseline sweep runs with the reused 10-case `1x` baseline for each method, giving 50 IDIR histories and 50 DirectCP 4-stage histories.

Progress is computed from smoothed `data_loss` curves. For each run or stage, progress is `(start - value_at_epoch) / (start - final_window_mean)`, so `1.0` means the curve has reached the final smoothed data-loss level observed in the completed run. This is an optimization-objective diagnostic, not a substitute for rerunning TRE at shorter schedules.

## Learning Rate Confirmation

- IDIR histories: `50`; unique recorded LR values: `[0.0001]`.
- DirectCP histories: `50`; unique recorded stage LR values: `[(0.001, 0.001, 0.001, 0.001)]`.
- DirectCP unique recorded stage epochs: `[(200, 400, 800, 1000)]`.

## Current Epoch Configuration

| Method | Epoch setting | LR | Default regularization |
| --- | --- | --- | --- |
| IDIR BE default | 2500 epochs | 1e-4 | alpha_bending=10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | stage epochs 200,400,800,1000; total 2400 | 1e-3 each stage | ptv_isotv_weight=0.11 |

## IDIR Observations

- IDIR curves are stochastic because each epoch samples 10,000 fixed-lung-mask coordinates.
- At the default `1x` factor, the mean smoothed data-loss progress is already about 99.1% at 1000 epochs and 99.6% at 1500 epochs.
- Low-regularization factors `0.01x` and `0.1x` converge more slowly; if those are being compared as final Pareto points, cutting too aggressively can bias them.

### IDIR Data-Loss Progress

| Factor | Candidate epoch | Mean progress | SD |
| --- | --- | --- | --- |
| 0.010 | 1000 | 0.981 | 0.011 |
| 0.010 | 1500 | 0.991 | 0.006 |
| 0.010 | 2000 | 0.996 | 0.002 |
| 0.010 | 2500 | 1.000 | 0.001 |
| 0.100 | 1000 | 0.986 | 0.006 |
| 0.100 | 1500 | 0.993 | 0.002 |
| 0.100 | 2000 | 0.997 | 0.001 |
| 0.100 | 2500 | 1.000 | 0.001 |
| 1.000 | 1000 | 0.991 | 0.002 |
| 1.000 | 1500 | 0.996 | 0.002 |
| 1.000 | 2000 | 0.998 | 0.001 |
| 1.000 | 2500 | 1.000 | 0.001 |
| 10.000 | 1000 | 0.993 | 0.003 |
| 10.000 | 1500 | 0.997 | 0.002 |
| 10.000 | 2000 | 0.999 | 0.002 |
| 10.000 | 2500 | 1.000 | 0.001 |
| 100.000 | 1000 | 0.997 | 0.002 |
| 100.000 | 1500 | 0.999 | 0.003 |
| 100.000 | 2000 | 1.000 | 0.003 |
| 100.000 | 2500 | 0.999 | 0.002 |

### IDIR Epoch To Reach Progress Threshold

| Factor | Threshold | Mean epoch | Median epoch | Max epoch | Cases |
| --- | --- | --- | --- | --- | --- |
| 0.100 | 0.990 | 1168.200 | 1181.000 | 1609.000 | 10 |
| 0.100 | 0.995 | 1616.300 | 1646.500 | 1861.000 | 10 |
| 0.100 | 0.999 | 2189.500 | 2215.000 | 2436.000 | 10 |
| 1.000 | 0.990 | 849.600 | 821.000 | 1079.000 | 10 |
| 1.000 | 0.995 | 1275.600 | 1240.500 | 1592.000 | 10 |
| 1.000 | 0.999 | 1950.400 | 1889.000 | 2299.000 | 10 |
| 10.000 | 0.990 | 689.300 | 643.500 | 1035.000 | 10 |
| 10.000 | 0.995 | 1053.200 | 1014.500 | 1326.000 | 10 |
| 10.000 | 0.999 | 1522.600 | 1543.500 | 1894.000 | 10 |

## DirectCP 4-Stage Observations

- Stage 0 should mostly stay at 200 epochs; it usually needs roughly 160-175 epochs to reach 99.9% of its final data-loss progress.
- Stage 1 is cheap, around 9% of stage optimization time at default `1x`; cutting it has limited wall-time benefit.
- At default `1x`, stages 2 and 3 are the main overtraining candidates. Stage 2 reaches 99.9% at about 89 epochs on average and at 218 epochs in the slowest case. Stage 3 reaches 99.9% at about 270 epochs on average and at 628 epochs in the slowest case.
- Extreme regularization factors have different behavior, especially stages 2-3. A shortened schedule should be labeled and validated if used for Pareto comparison.

### DirectCP Default-Factor Data-Loss Progress

| Stage | Candidate epoch | Mean progress | SD |
| --- | --- | --- | --- |
| 0 | 150 | 0.972 | 0.018 |
| 0 | 200 | 1.017 | 0.014 |
| 1 | 200 | 0.998 | 0.005 |
| 1 | 300 | 0.999 | 0.002 |
| 1 | 400 | 1.000 | 0.000 |
| 2 | 200 | 1.000 | 0.001 |
| 2 | 400 | 1.000 | 0.000 |
| 2 | 800 | 1.000 | 0.000 |
| 3 | 250 | 0.999 | 0.001 |
| 3 | 500 | 0.999 | 0.001 |
| 3 | 750 | 1.000 | 0.000 |
| 3 | 1000 | 1.000 | 0.000 |

### DirectCP Default-Factor Epoch To Reach Progress Threshold

| Stage | Threshold | Mean epoch | Median epoch | Max epoch | Cases |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.990 | 159.300 | 161.500 | 171.000 | 10 |
| 0 | 0.995 | 165.100 | 166.500 | 173.000 | 10 |
| 0 | 0.999 | 171.600 | 172.000 | 175.000 | 10 |
| 1 | 0.990 | 98.900 | 80.500 | 268.000 | 10 |
| 1 | 0.995 | 115.400 | 93.000 | 300.000 | 10 |
| 1 | 0.999 | 156.300 | 124.500 | 359.000 | 10 |
| 2 | 0.990 | 32.700 | 27.000 | 62.000 | 10 |
| 2 | 0.995 | 38.500 | 31.000 | 78.000 | 10 |
| 2 | 0.999 | 89.000 | 75.000 | 218.000 | 10 |
| 3 | 0.990 | 69.900 | 79.000 | 142.000 | 10 |
| 3 | 0.995 | 104.000 | 110.000 | 205.000 | 10 |
| 3 | 0.999 | 269.600 | 223.500 | 628.000 | 10 |

### DirectCP Default-Factor Stage Timing

| Stage | Elapsed sec | Sec/epoch | Stage-time fraction |
| --- | --- | --- | --- |
| 0 | 47.75 +/- 10.15 | 0.2388 +/- 0.0507 | 0.361 +/- 0.017 |
| 1 | 11.63 +/- 1.63 | 0.0291 +/- 0.0041 | 0.089 +/- 0.008 |
| 2 | 25.33 +/- 3.49 | 0.0317 +/- 0.0044 | 0.194 +/- 0.017 |
| 3 | 46.96 +/- 7.92 | 0.0470 +/- 0.0079 | 0.357 +/- 0.017 |

## Epoch-Reduction Recommendation

For final paper-facing reproduction, do not change epochs silently. Any shortened schedule should be a new protocol tag and should be validated by rerunning TRE/regularity for at least the default `1x` point and the neighboring Pareto-relevant factors.

Recommended next validation runs:

- IDIR default/near-default validation: compare `epochs=1500`, `2000`, and current `2500` for factors `0.1,1,10` on all 10 cases. The objective curves support `1500` as a likely screening setting and `2000` as a safer final-lite setting.
- DirectCP default validation: compare current `200,400,800,1000` against `200,400,300,650` and fast-screen `200,200,200,400` at factor `1x` on all 10 cases. The curve evidence favors cutting stages 2-3 first, not stage 0.
- If the shortened DirectCP default point matches original-snap TRE within an acceptable tolerance, extend only to the Pareto-relevant neighboring factors `0.1x` and `10x`. Avoid using the shortened schedule for extreme `0.01x`/`100x` conclusions without separate validation.

Practical default recommendation before validation:

- IDIR screen: `1500` epochs; IDIR safer final-lite: `2000` epochs.
- DirectCP screen: `200,200,200,400`; DirectCP safer final-lite: `200,400,300,650`.
- Keep the completed `2500` and `200,400,800,1000` rows as the current canonical reproduction until shorter schedules are empirically checked with TRE.

## Plots

- `plots/training_curves/idir_data_loss_all_cases_by_factor.png`
- `plots/training_curves/idir_mean_progress_by_factor.png`
- `plots/training_curves/directcp_progress_factor_*.png`
- `plots/training_curves/directcp_mean_progress_by_factor_and_stage.png`
