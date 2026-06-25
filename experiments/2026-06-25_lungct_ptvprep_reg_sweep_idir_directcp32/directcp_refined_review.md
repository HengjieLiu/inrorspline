# LungCT DirectCP Refined Regularization Figures

This supplemental report adds DirectCP 32-16-8-4 factors `0.2x`, `0.5x`, `2x`, and `5x` to the original combined ellipse-style ARC/Pareto figures. Original figures under `plots/` are not overwritten; regenerated refined figures are under `plots_refined_directcp/`.

IDIR is shown at the original coarse factors `0.01x`, `0.1x`, `1x`, `10x`, and `100x`. DirectCP is shown at `0.01x`, `0.1x`, `0.2x`, `0.5x`, `1x`, `2x`, `5x`, `10x`, and `100x`.

Combined plot ellipses are axis-aligned 1-SD ellipses: horizontal radius is one standard deviation of the x-axis regularity metric across cases and vertical radius is one standard deviation of original-snap TRE across cases.

## Mean Rows

| Method | Factor | Reg value | Cases | Original Snap | Original Sub | 1mm Snap | 1mm Sub | FG logJ std | FG J<=0% | BE fg | Diff fg | Disp mean | Disp p95 | Disp max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.01 | 0.001100 | 10 | 1.320 +/- 0.713 | 1.393 +/- 0.704 | 1.503 +/- 0.712 | 1.396 +/- 0.710 | 0.661347 | 2.6170 | 0.132987 | 0.098929 | 8.932 | 18.890 | 30.600 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.10 | 0.011000 | 10 | 1.022 +/- 0.206 | 1.121 +/- 0.169 | 1.242 +/- 0.162 | 1.122 +/- 0.168 | 0.397449 | 0.4137 | 0.031837 | 0.036033 | 8.961 | 18.905 | 25.836 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.20 | 0.022000 | 10 | 0.951 +/- 0.160 | 1.059 +/- 0.131 | 1.172 +/- 0.116 | 1.059 +/- 0.129 | 0.310308 | 0.1316 | 0.015665 | 0.023239 | 8.955 | 18.881 | 24.828 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.50 | 0.055000 | 10 | 0.917 +/- 0.148 | 1.048 +/- 0.124 | 1.162 +/- 0.114 | 1.049 +/- 0.122 | 0.223834 | 0.0367 | 0.005517 | 0.014009 | 8.915 | 18.758 | 24.058 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 0.110000 | 10 | 0.922 +/- 0.152 | 1.049 +/- 0.119 | 1.168 +/- 0.114 | 1.052 +/- 0.118 | 0.179961 | 0.0138 | 0.002527 | 0.010327 | 8.880 | 18.639 | 23.548 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 2.00 | 0.220000 | 10 | 0.937 +/- 0.149 | 1.061 +/- 0.116 | 1.182 +/- 0.112 | 1.065 +/- 0.114 | 0.148482 | 0.0027 | 0.001241 | 0.008023 | 8.833 | 18.445 | 22.759 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 5.00 | 0.550000 | 10 | 0.978 +/- 0.159 | 1.104 +/- 0.114 | 1.227 +/- 0.110 | 1.113 +/- 0.114 | 0.116674 | 0.0000 | 0.000508 | 0.005895 | 8.716 | 17.917 | 21.467 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10.00 | 1.100000 | 10 | 1.583 +/- 1.289 | 1.704 +/- 1.258 | 1.827 +/- 1.268 | 1.727 +/- 1.279 | 0.094587 | 0.0000 | 0.000192 | 0.003711 | 8.024 | 15.445 | 18.413 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 100.00 | 11.000000 | 10 | 6.809 +/- 3.425 | 6.581 +/- 3.310 | 6.669 +/- 3.328 | 6.635 +/- 3.337 | 0.011861 | 0.0000 | 0.000000 | 0.000030 | 2.618 | 2.996 | 3.121 |
| IDIR BE default | 0.01 | 0.100000 | 10 | 1.317 +/- 0.656 | 1.408 +/- 0.634 | 1.513 +/- 0.626 | 1.413 +/- 0.639 | 0.322472 | 0.2938 | 0.007477 | 0.023005 | 8.715 | 18.574 | 23.950 |
| IDIR BE default | 0.10 | 1.000000 | 10 | 1.046 +/- 0.200 | 1.159 +/- 0.166 | 1.264 +/- 0.162 | 1.161 +/- 0.167 | 0.215510 | 0.0226 | 0.002427 | 0.012305 | 8.876 | 18.808 | 23.648 |
| IDIR BE default | 1.00 | 10.000000 | 10 | 1.017 +/- 0.176 | 1.133 +/- 0.137 | 1.236 +/- 0.133 | 1.138 +/- 0.136 | 0.142869 | 0.0000 | 0.000535 | 0.007218 | 8.893 | 18.850 | 23.105 |
| IDIR BE default | 10.00 | 100.000000 | 10 | 1.085 +/- 0.212 | 1.196 +/- 0.163 | 1.294 +/- 0.156 | 1.202 +/- 0.163 | 0.104202 | 0.0000 | 0.000115 | 0.005160 | 8.828 | 18.824 | 23.152 |
| IDIR BE default | 100.00 | 1000.000000 | 10 | 1.257 +/- 0.269 | 1.333 +/- 0.213 | 1.435 +/- 0.207 | 1.344 +/- 0.217 | 0.078938 | 0.0000 | 0.000020 | 0.003827 | 8.724 | 18.510 | 22.526 |

## Case Counts

| Method | Factor | Cases |
| --- | --- | --- |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.01 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.10 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.20 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 0.50 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 1.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 2.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 5.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 10.00 | 10 |
| DirectCP 32-16-8-4 pTV LCC + isoTV | 100.00 | 10 |
| IDIR BE default | 0.01 | 10 |
| IDIR BE default | 0.10 | 10 |
| IDIR BE default | 1.00 | 10 |
| IDIR BE default | 10.00 | 10 |
| IDIR BE default | 100.00 | 10 |

## Artifacts

- Refined plots: `plots_refined_directcp/`.
- Tables: `directcp_refined_case_rows.csv/json`, `directcp_refined_mean_rows.csv/json`, and `directcp_refined_factor_counts.csv`.
