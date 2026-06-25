---
doc_type: technical_log
status: completed
date_started: 2026-06-25
date_completed: 2026-06-25
---

# Technical Log

## Known Limitations

- This sweep isolates regularization weight only. Learning rates, epochs, stage
  schedule, similarity loss, and preprocessing remain fixed.
- IDIR bending and DirectCP isoTV weights are not on the same absolute scale.
  The sweep is therefore expressed as a multiplicative factor around each
  method's own default.
- The canonical regenerated pTVReg crop uses the corrected pTVReg/MATLAB-origin
  convention, not the old task4-compatible `+1` shifted crop.
- `original_snap` is the primary comparison metric. The other three TRE columns
  are retained for audit and metric-domain diagnostics.

## Verification Checklist

- Launcher dry-run: passed. Exactly 80 nonbaseline jobs were queued; no
  `1e0` baseline jobs were queued.
- Smoke jobs: passed. Case `01` completed for both methods at factors `0.01`
  and `100` with minimal epochs and correct regularization values in
  `summary.json`.
- Full sweep launch: passed. 80/80 nonbaseline jobs completed with return code
  0.
- Aggregation and Pareto plots: passed. Aggregation merged the 80 sweep rows
  with 20 reused baseline rows, producing 100 case rows, 10 mean rows, 8
  method-specific Pareto plots, and 4 combined method-comparison Pareto plots.

## Outputs

- Sweep output root:
  `/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1`.
- Baseline root:
  `/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1`.
- Review report:
  `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/review_report.md`.
- Tables: `sweep_case_rows.csv/json` and `sweep_mean_rows.csv/json`.
- Pareto plots: method-specific `plots/*original_snap_vs*.png` plus combined
  `plots/combined_original_snap_vs*.png`.

Combined plot convention: each shaded ellipse is centered at the method/factor
mean point. The horizontal radius is one standard deviation of the plotted
regularity metric across the 10 cases, and the vertical radius is one standard
deviation of `original_snap` TRE across the 10 cases. Points are connected in
increasing regularization-factor order: `0.01x`, `0.1x`, `1x`, `10x`, `100x`.

## Compute Cost And Epoch Record

Epoch settings:

| Method | Epoch setting | LR | Schedule | Default regularization |
| --- | --- | --- | --- | --- |
| IDIR BE default | 2500 epochs | 1e-4 | single stage | alpha_bending=10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | stage epochs 200,400,800,1000 (total 2400) | 1e-3 | CPS 32,16,8,4; downsample 8,4,2,1 | ptv_isotv_weight=0.11 |

Default `1x` per-case cost:

| Case | IDIR time min | IDIR peak GB | DirectCP time min | DirectCP peak GB |
| --- | --- | --- | --- | --- |
| 01 | 2.48 | 4.07 | 3.71 | 3.96 |
| 02 | 2.74 | 4.25 | 5.84 | 7.45 |
| 03 | 2.51 | 4.18 | 3.81 | 6.13 |
| 04 | 2.66 | 4.13 | 3.81 | 5.42 |
| 05 | 2.91 | 4.15 | 3.70 | 5.71 |
| 06 | 2.53 | 4.13 | 3.83 | 5.26 |
| 07 | 2.61 | 4.15 | 3.61 | 5.78 |
| 08 | 2.44 | 4.21 | 4.24 | 6.49 |
| 09 | 2.47 | 4.05 | 3.03 | 3.64 |
| 10 | 2.92 | 4.12 | 3.43 | 4.99 |

Default `1x` summary: IDIR `2.63 +/- 0.17` min and `4.14 +/- 0.06`
GB peak memory; DirectCP 4-stage `3.90 +/- 0.71` min and `5.48 +/- 1.07`
GB peak memory.

All-factor summary: IDIR `2.71 +/- 0.17` min and `4.14 +/- 0.06` GB
peak memory across 50 rows; DirectCP 4-stage `4.18 +/- 0.65` min and
`5.48 +/- 1.07` GB peak memory across 50 rows.

## Training-Curve Analysis

Analysis artifact:
`experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/training_curve_analysis/training_curve_epoch_analysis.md`.

Plots:
`experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/plots/training_curves/`.

Learning-rate confirmation:

- IDIR: 50/50 planned histories record `lr=1e-4`.
- DirectCP 4-stage: 50/50 planned histories record stage LR
  `[1e-3,1e-3,1e-3,1e-3]`.
- DirectCP 4-stage histories record stage epochs `[200,400,800,1000]`.

Curve-based epoch-reduction observations:

- IDIR default `1x` reaches about `99.1%` mean smoothed data-loss progress by
  epoch `1000` and `99.6%` by epoch `1500`. Across default cases, the mean
  epoch to reach `99.5%` progress is `1276`, with slowest case `1592`.
- Low IDIR regularization factors `0.01x` and `0.1x` converge more slowly, so
  aggressive epoch cuts could bias those endpoints if they are used as final
  Pareto points.
- DirectCP default `1x` stage 0 should mostly stay near `200` epochs; it
  reaches `99.9%` progress at mean `172` stage epochs, slowest `175`.
- DirectCP default `1x` stages 2 and 3 are the main overtraining candidates:
  stage 2 reaches `99.9%` progress at mean `89`, slowest `218`; stage 3 reaches
  `99.9%` progress at mean `270`, slowest `628`.

Recommended validation, not yet launched:

- IDIR: compare `1500`, `2000`, and current `2500` epochs for factors
  `0.1,1,10` on all 10 cases. Use `1500` as a likely screening setting and
  `2000` as a safer final-lite setting if TRE/regularity match.
- DirectCP: compare current `200,400,800,1000` against safer final-lite
  `200,400,300,650` and fast-screen `200,200,200,400` at factor `1x` on all 10
  cases. If the default point matches, extend only to neighboring factors
  `0.1x` and `10x` before using shortened schedules in a Pareto comparison.

## Result Summary

`original_snap` TRE is the primary reporting/comparison metric.

| Method | Factor | Reg value | Original snap TRE | FG logJ std | FG J<=0% | BE fg | Diff fg | Disp mean | Disp p95 | Disp max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR BE default | 0.01 | 0.1 | 1.317 +/- 0.656 | 0.322472 | 0.2938 | 0.007477 | 0.023005 | 8.715 | 18.574 | 23.950 |
| IDIR BE default | 0.10 | 1 | 1.046 +/- 0.200 | 0.215510 | 0.0226 | 0.002427 | 0.012305 | 8.876 | 18.808 | 23.648 |
| IDIR BE default | 1.00 | 10 | 1.017 +/- 0.176 | 0.142869 | 0.0000 | 0.000535 | 0.007218 | 8.893 | 18.850 | 23.105 |
| IDIR BE default | 10.00 | 100 | 1.085 +/- 0.212 | 0.104202 | 0.0000 | 0.000115 | 0.005160 | 8.828 | 18.824 | 23.152 |
| IDIR BE default | 100.00 | 1000 | 1.257 +/- 0.269 | 0.078938 | 0.0000 | 0.000020 | 0.003827 | 8.724 | 18.510 | 22.526 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.01 | 0.0011 | 1.320 +/- 0.713 | 0.661347 | 2.6170 | 0.132987 | 0.098929 | 8.932 | 18.890 | 30.600 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.10 | 0.011 | 1.022 +/- 0.206 | 0.397449 | 0.4137 | 0.031837 | 0.036033 | 8.961 | 18.905 | 25.836 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 0.11 | 0.922 +/- 0.152 | 0.179961 | 0.0138 | 0.002527 | 0.010327 | 8.880 | 18.639 | 23.548 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10.00 | 1.1 | 1.583 +/- 1.289 | 0.094587 | 0.0000 | 0.000192 | 0.003711 | 8.024 | 15.445 | 18.413 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 100.00 | 11 | 6.809 +/- 3.425 | 0.011861 | 0.0000 | 0.000000 | 0.000030 | 2.618 | 2.996 | 3.121 |

Best `original_snap` points were the default `1e0` regularization for both
methods: IDIR `alpha_bending=10` and DirectCP `ptv_isotv_weight=0.11`.

## DirectCP Refined-Factor Addendum

Requested after the coarse sweep: add DirectCP 32-16-8-4 factors `0.2x`,
`0.5x`, `2x`, and `5x` on cases `01-10`, using GPUs `0,1`, without
overwriting the original figures.

- Launch result: 40/40 DirectCP refinement jobs completed with return code 0.
- New DirectCP isoTV weights: `0.022`, `0.055`, `0.22`, and `0.55`.
- Aggregation result: 140 case rows and 14 mean rows in the refined comparison
  set: IDIR coarse factors plus DirectCP factors
  `0.01,0.1,0.2,0.5,1,2,5,10,100`.
- New figures: `plots_refined_directcp/`; original figures under `plots/` were
  not overwritten.
- Supplemental report: `directcp_refined_review.md`.

DirectCP refined original-snap TRE and foreground regularity:

| Factor | isoTV weight | Original snap TRE | FG logJ std | FG J<=0% | BE fg | Diff fg |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.0011 | 1.320 +/- 0.713 | 0.661347 | 2.6170 | 0.132987 | 0.098929 |
| 0.10 | 0.011 | 1.022 +/- 0.206 | 0.397449 | 0.4137 | 0.031837 | 0.036033 |
| 0.20 | 0.022 | 0.951 +/- 0.160 | 0.310308 | 0.1316 | 0.015665 | 0.023239 |
| 0.50 | 0.055 | 0.917 +/- 0.148 | 0.223834 | 0.0367 | 0.005517 | 0.014009 |
| 1.00 | 0.11 | 0.922 +/- 0.152 | 0.179961 | 0.0138 | 0.002527 | 0.010327 |
| 2.00 | 0.22 | 0.937 +/- 0.149 | 0.148482 | 0.0027 | 0.001241 | 0.008023 |
| 5.00 | 0.55 | 0.978 +/- 0.159 | 0.116674 | 0.0000 | 0.000508 | 0.005895 |
| 10.00 | 1.1 | 1.583 +/- 1.289 | 0.094587 | 0.0000 | 0.000192 | 0.003711 |
| 100.00 | 11 | 6.809 +/- 3.425 | 0.011861 | 0.0000 | 0.000000 | 0.000030 |

Refined conclusion: DirectCP `0.5x` (`ptv_isotv_weight=0.055`) is the best
mean original-snap TRE point in this grid, but it is only marginally better than
the original `1x` point (`0.917` vs `0.922` mm). Factors `2x` and `5x` improve
regularity with modest TRE cost; `10x` and especially `100x` are too strong for
accuracy.

## IDIR Refined-Factor Addendum

Requested after the DirectCP refinement: add IDIR BE factors `1/30x`, `1/3x`,
and `3x` on cases `01-10`, using GPUs `1,2,3`, and regenerate the ROI combined
ellipse-style figures alongside `plots_refined_directcp_roi_no_extremes/`.

- Launch result: 30/30 IDIR refinement jobs completed with return code 0.
- New IDIR bending weights: `0.333333`, `3.333333`, and `30`.
- Aggregation result: 150 case rows and 15 mean rows in the refined ROI
  comparison set: IDIR factors `0.01,0.033333,0.1,0.333333,1,3,10,100` plus
  DirectCP factors `0.1,0.2,0.5,1,2,5,10`.
- New figures: `plots_refined_both_roi_no_extremes/`; the existing
  `plots_refined_directcp_roi_no_extremes/` figures were not overwritten.
- Supplemental report: `both_refined_roi_no_extremes_review.md`.

IDIR refined original-snap TRE and foreground regularity:

| Factor | alpha_bending | Original snap TRE | FG logJ std | FG J<=0% | BE fg | Diff fg |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 0.1 | 1.317 +/- 0.656 | 0.322472 | 0.2938 | 0.007477 | 0.023005 |
| 0.033333 | 0.333333 | 1.244 +/- 0.491 | 0.275920 | 0.1440 | 0.004979 | 0.018156 |
| 0.10 | 1 | 1.046 +/- 0.200 | 0.215510 | 0.0226 | 0.002427 | 0.012305 |
| 0.333333 | 3.333333 | 1.028 +/- 0.179 | 0.172573 | 0.0063 | 0.001119 | 0.008981 |
| 1.00 | 10 | 1.017 +/- 0.176 | 0.142869 | 0.0000 | 0.000535 | 0.007218 |
| 3.00 | 30 | 1.037 +/- 0.180 | 0.121340 | 0.0000 | 0.000259 | 0.006062 |
| 10.00 | 100 | 1.085 +/- 0.212 | 0.104202 | 0.0000 | 0.000115 | 0.005160 |
| 100.00 | 1000 | 1.257 +/- 0.269 | 0.078938 | 0.0000 | 0.000020 | 0.003827 |

Refined conclusion: IDIR `1x` (`alpha_bending=10`) remains the best mean
original-snap TRE point in this grid. The `1/3x` and `3x` points are close but
do not improve the primary metric; `3x` provides smoother regularity with a
small TRE cost.
