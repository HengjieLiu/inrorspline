# LungCT Corrected MR-D-BSCP IDIR-Loss Test

## Protocol

- Method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`.
- Schedule: `32,16,8,4`; downsample `8,4,2,1`; stage epochs `200,400,800,1000`.
- Alpha: `10000`; LR `1e-3`; seed `1`; sampled batch size `10000`.
- Loss: fixed T00 lung-mask sampled global NCC plus sampled analytic cubic B-spline BE.
- Inputs: canonical pTVReg-style 1mm landmark crop, not lung-masked images.
- Metrics: original/native-crop snap TRE and fixed T00 lung-mask foreground regularity.

## Completion

- Completed rows: `10`.
- Completion errors: `[]`.

## Mean Comparison

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- |
| Current FD-BE | 0.966 +/- 0.151 | 0.185452 +/- 0.060879 | 0.0037 | 0.001971 | 0.010033 |
| Corrected analytic-mask | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 | 0.006341 | 0.016738 |

## Corrected Case Rows

| Case | TRE | FG stdLogJ | FG J<=0 % | Elapsed sec | Peak GB |
| --- | --- | --- | --- | --- | --- |
| 01 | 0.918 | 0.174376 | 0.0000 | 770.6 | 3.97 |
| 02 | 0.883 | 0.192221 | 0.0060 | 815.5 | 7.46 |
| 03 | 1.031 | 0.199060 | 0.0080 | 766.2 | 6.14 |
| 04 | 1.366 | 0.279349 | 0.0005 | 796.1 | 5.43 |
| 05 | 1.188 | 0.267399 | 0.0145 | 847.6 | 5.72 |
| 06 | 1.000 | 0.397272 | 0.0791 | 757.4 | 5.27 |
| 07 | 0.905 | 0.304141 | 0.0262 | 791.4 | 5.79 |
| 08 | 1.184 | 0.320837 | 0.1954 | 771.8 | 6.50 |
| 09 | 1.020 | 0.252155 | 0.0049 | 723.1 | 3.64 |
| 10 | 0.878 | 0.267143 | 0.0141 | 810.8 | 5.00 |

## Artifacts

- `case_rows.csv`, `case_rows.json`
- `mean_row.csv`, `mean_row.json`
- `comparison_rows.csv`, `comparison_rows.json`
