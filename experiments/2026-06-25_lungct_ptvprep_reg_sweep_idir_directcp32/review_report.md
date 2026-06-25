# LungCT IDIR + DirectCP 32-16-8-4 Regularization Sweep

Original Snap TRE is the primary comparison metric. Original means native/crop-space TRE after resizing the saved 1mm displacement field back to the corrected pTVReg native landmark crop. The three other TRE columns are retained as diagnostics.

Combined Pareto plots include translucent axis-aligned ellipses at each factor point. Each ellipse is centered at the mean point, with horizontal radius equal to one standard deviation of the x-axis regularity metric across cases and vertical radius equal to one standard deviation of original-snap TRE across cases. Points are connected in increasing regularization-factor order: 0.01x, 0.1x, 1x, 10x, 100x.

## Mean Rows

| Method | Factor | Reg value | Cases | Original Snap | Original Sub | 1mm Snap | 1mm Sub | FG logJ std | FG J<=0% | BE fg | Diff fg | Disp mean | Disp p95 | Disp max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.01 | 0.001100 | 10 | 1.320 +/- 0.713 | 1.393 +/- 0.704 | 1.503 +/- 0.712 | 1.396 +/- 0.710 | 0.661347 | 2.6170 | 0.132987 | 0.098929 | 8.932 | 18.890 | 30.600 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.10 | 0.011000 | 10 | 1.022 +/- 0.206 | 1.121 +/- 0.169 | 1.242 +/- 0.162 | 1.122 +/- 0.168 | 0.397449 | 0.4137 | 0.031837 | 0.036033 | 8.961 | 18.905 | 25.836 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 0.110000 | 10 | 0.922 +/- 0.152 | 1.049 +/- 0.119 | 1.168 +/- 0.114 | 1.052 +/- 0.118 | 0.179961 | 0.0138 | 0.002527 | 0.010327 | 8.880 | 18.639 | 23.548 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10.00 | 1.100000 | 10 | 1.583 +/- 1.289 | 1.704 +/- 1.258 | 1.827 +/- 1.268 | 1.727 +/- 1.279 | 0.094587 | 0.0000 | 0.000192 | 0.003711 | 8.024 | 15.445 | 18.413 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 100.00 | 11.000000 | 10 | 6.809 +/- 3.425 | 6.581 +/- 3.310 | 6.669 +/- 3.328 | 6.635 +/- 3.337 | 0.011861 | 0.0000 | 0.000000 | 0.000030 | 2.618 | 2.996 | 3.121 |
| IDIR BE default | 0.01 | 0.100000 | 10 | 1.317 +/- 0.656 | 1.408 +/- 0.634 | 1.513 +/- 0.626 | 1.413 +/- 0.639 | 0.322472 | 0.2938 | 0.007477 | 0.023005 | 8.715 | 18.574 | 23.950 |
| IDIR BE default | 0.10 | 1.000000 | 10 | 1.046 +/- 0.200 | 1.159 +/- 0.166 | 1.264 +/- 0.162 | 1.161 +/- 0.167 | 0.215510 | 0.0226 | 0.002427 | 0.012305 | 8.876 | 18.808 | 23.648 |
| IDIR BE default | 1.00 | 10.000000 | 10 | 1.017 +/- 0.176 | 1.133 +/- 0.137 | 1.236 +/- 0.133 | 1.138 +/- 0.136 | 0.142869 | 0.0000 | 0.000535 | 0.007218 | 8.893 | 18.850 | 23.105 |
| IDIR BE default | 10.00 | 100.000000 | 10 | 1.085 +/- 0.212 | 1.196 +/- 0.163 | 1.294 +/- 0.156 | 1.202 +/- 0.163 | 0.104202 | 0.0000 | 0.000115 | 0.005160 | 8.828 | 18.824 | 23.152 |
| IDIR BE default | 100.00 | 1000.000000 | 10 | 1.257 +/- 0.269 | 1.333 +/- 0.213 | 1.435 +/- 0.207 | 1.344 +/- 0.217 | 0.078938 | 0.0000 | 0.000020 | 0.003827 | 8.724 | 18.510 | 22.526 |

## Best Original Snap By Method

| Method | Best factor | Reg value | Original Snap | FG logJ std | FG J<=0% | BE fg | Diff fg | Disp mean | Disp p95 | Disp max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR BE default | 1.00 | 10.000000 | 1.017 | 0.142869 | 0.0000 | 0.000535 | 0.007218 | 8.893 | 18.850 | 23.105 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 0.110000 | 0.922 | 0.179961 | 0.0138 | 0.002527 | 0.010327 | 8.880 | 18.639 | 23.548 |

## Compute Cost And Epochs

Runtime and memory are from completed `summary.json` files. The per-case table below uses the reused default `1x` baseline rows, because those are the primary comparison points. The all-factor summary includes all 5 factors per method.

### Epoch Settings

| Method | Epoch setting | LR | Schedule | Default regularization |
| --- | --- | --- | --- | --- |
| IDIR BE default | 2500 epochs | 1e-4 | single stage | alpha_bending=10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | stage epochs 200,400,800,1000 (total 2400) | 1e-3 | CPS 32,16,8,4; downsample 8,4,2,1 | ptv_isotv_weight=0.11 |

### Default 1x Per-Case Cost

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

### Default 1x Summary

| Method | Rows (default 1x) | Elapsed min mean +/- SD | Elapsed min range | Peak GB mean +/- SD | Peak GB range |
| --- | --- | --- | --- | --- | --- |
| IDIR BE default | 10 | 2.63 +/- 0.17 | 2.44-2.92 | 4.14 +/- 0.06 | 4.05-4.25 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10 | 3.90 +/- 0.71 | 3.03-5.84 | 5.48 +/- 1.07 | 3.64-7.45 |

### All-Factor Summary

| Method | Rows (all factors) | Elapsed min mean +/- SD | Elapsed min range | Peak GB mean +/- SD | Peak GB range |
| --- | --- | --- | --- | --- | --- |
| IDIR BE default | 50 | 2.71 +/- 0.17 | 2.42-3.07 | 4.14 +/- 0.06 | 4.05-4.25 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 50 | 4.18 +/- 0.65 | 3.03-6.13 | 5.48 +/- 1.07 | 3.64-7.45 |

## Case Counts

| Method | Factor | Cases |
| --- | --- | --- |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.01 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.10 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 100.00 | 10 |
| IDIR BE default | 0.01 | 10 |
| IDIR BE default | 0.10 | 10 |
| IDIR BE default | 1.00 | 10 |
| IDIR BE default | 10.00 | 10 |
| IDIR BE default | 100.00 | 10 |

## Artifacts

- Per-case rows: `sweep_case_rows.csv` and `sweep_case_rows.json`.
- Mean rows: `sweep_mean_rows.csv` and `sweep_mean_rows.json`.
- Pareto plots: method-specific `plots/*original_snap_vs*.png` and combined `plots/combined_original_snap_vs*.png`.
- Training-curve analysis: `training_curve_analysis/training_curve_epoch_analysis.md` and `plots/training_curves/`.
