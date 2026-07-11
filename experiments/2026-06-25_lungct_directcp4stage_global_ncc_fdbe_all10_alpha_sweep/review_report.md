# LungCT DirectCP 4-stage Global NCC + Dense FD-BE All-Case Sweep

## Summary

- FD-BE method: `direct_cp_32_16_8_4_global_ncc_fd_be`.
- Stage schedule: `32,16,8,4`; downsample schedule: `8,4,2,1`; stage epochs: `200,400,800,1000`.
- FD-BE alpha grid: `0,10,100,1000,10000,100000`.
- FD-BE rows merge existing cases `01,08` with new missing-case runs.
- Comparison figures use all 10 cases for IDIR, pTV DirectCP, and FD-BE.
- Ellipses are axis-aligned 1-SD ellipses across cases.
- Ellipse centers are method/reg-weight means; horizontal radius is the across-case SD of the plotted regularity metric, vertical radius is the across-case SD of original-snap TRE, and covariance is not used.

## Completion

- FD-BE rows: `60` expected `60`.
- Comparison rows: `160`.
- Completion check: passed.

## Best Mean Points

- IDIR BE: best `a=10` with original snap TRE `1.017 +/- 0.176 mm`, FG stdLogJ `0.142869`, FG J<=0 `0.0000%`.
- DirectCP 4-stage pTV LCC + isoTV: best `w=0.11` with original snap TRE `0.922 +/- 0.152 mm`, FG stdLogJ `0.179961`, FG J<=0 `0.0138%`.
- DirectCP 4-stage global NCC + FD-BE: best `a=10000` with original snap TRE `0.966 +/- 0.151 mm`, FG stdLogJ `0.185452`, FG J<=0 `0.0037%`.

## FD-BE Mean Rows

| Reg | Original snap TRE | Original sub TRE | FG stdLogJ | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- | --- |
| a=0 | 2.096 +/- 1.698 | 2.149 +/- 1.679 | 0.971325 | 10.7841 | 0.345415 | 0.270999 |
| a=10 | 1.897 +/- 1.475 | 1.968 +/- 1.470 | 0.879791 | 7.2790 | 0.220361 | 0.185664 |
| a=100 | 1.364 +/- 0.588 | 1.439 +/- 0.573 | 0.638078 | 2.2930 | 0.074157 | 0.079324 |
| a=1000 | 1.017 +/- 0.155 | 1.120 +/- 0.137 | 0.347456 | 0.2440 | 0.013814 | 0.025627 |
| a=10000 | 0.966 +/- 0.151 | 1.101 +/- 0.124 | 0.185452 | 0.0037 | 0.001971 | 0.010033 |
| a=100000 | 1.013 +/- 0.187 | 1.150 +/- 0.141 | 0.118115 | 0.0000 | 0.000299 | 0.006244 |

## Comparison Mean Rows


### IDIR BE

| Reg | Original snap TRE | Original sub TRE | FG stdLogJ | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- | --- |
| a=0.1 | 1.317 +/- 0.656 | 1.408 +/- 0.634 | 0.322472 | 0.2938 | 0.007477 | 0.023005 |
| a=1 | 1.046 +/- 0.200 | 1.159 +/- 0.166 | 0.215510 | 0.0226 | 0.002427 | 0.012305 |
| a=10 | 1.017 +/- 0.176 | 1.133 +/- 0.137 | 0.142869 | 0.0000 | 0.000535 | 0.007218 |
| a=100 | 1.085 +/- 0.212 | 1.196 +/- 0.163 | 0.104202 | 0.0000 | 0.000115 | 0.005160 |
| a=1000 | 1.257 +/- 0.269 | 1.333 +/- 0.213 | 0.078938 | 0.0000 | 0.000020 | 0.003827 |

### DirectCP 4-stage pTV LCC + isoTV

| Reg | Original snap TRE | Original sub TRE | FG stdLogJ | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- | --- |
| w=0.0011 | 1.320 +/- 0.713 | 1.393 +/- 0.704 | 0.661347 | 2.6170 | 0.132987 | 0.098929 |
| w=0.011 | 1.022 +/- 0.206 | 1.121 +/- 0.169 | 0.397449 | 0.4137 | 0.031837 | 0.036033 |
| w=0.11 | 0.922 +/- 0.152 | 1.049 +/- 0.119 | 0.179961 | 0.0138 | 0.002527 | 0.010327 |
| w=1.1 | 1.583 +/- 1.289 | 1.704 +/- 1.258 | 0.094587 | 0.0000 | 0.000192 | 0.003711 |
| w=11 | 6.809 +/- 3.425 | 6.581 +/- 3.310 | 0.011861 | 0.0000 | 0.000000 | 0.000030 |

### DirectCP 4-stage global NCC + FD-BE

| Reg | Original snap TRE | Original sub TRE | FG stdLogJ | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- | --- |
| a=0 | 2.096 +/- 1.698 | 2.149 +/- 1.679 | 0.971325 | 10.7841 | 0.345415 | 0.270999 |
| a=10 | 1.897 +/- 1.475 | 1.968 +/- 1.470 | 0.879791 | 7.2790 | 0.220361 | 0.185664 |
| a=100 | 1.364 +/- 0.588 | 1.439 +/- 0.573 | 0.638078 | 2.2930 | 0.074157 | 0.079324 |
| a=1000 | 1.017 +/- 0.155 | 1.120 +/- 0.137 | 0.347456 | 0.2440 | 0.013814 | 0.025627 |
| a=10000 | 0.966 +/- 0.151 | 1.101 +/- 0.124 | 0.185452 | 0.0037 | 0.001971 | 0.010033 |
| a=100000 | 1.013 +/- 0.187 | 1.150 +/- 0.141 | 0.118115 | 0.0000 | 0.000299 | 0.006244 |

## Artifacts

- FD-BE CSV/JSON: `fdbe_case_rows.*`, `fdbe_mean_rows.*`.
- Comparison CSV/JSON: `comparison_case_rows.*`, `comparison_mean_rows.*`.
- Ellipse plots: `plots_compare_ellipses/all10_original_snap_vs_*_ellipses.png`.
