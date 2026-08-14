# LungCT Corrected MR-D-BSCP IDIR-Loss Alpha Sweep

## Protocol

- Method: `direct_cp_32_16_8_4_masked_ncc_bspline_be`.
- Schedule: `32,16,8,4`; downsample `8,4,2,1`; stage epochs `200,400,800,1000`.
- LR `1e-3`; seed `1`; sampled batch size `10000`.
- Loss: fixed T00 lung-mask sampled global NCC plus sampled analytic cubic B-spline BE.
- Inputs: canonical pTVReg-style 1mm landmark crop, not lung-masked images.
- Metrics: original/native-crop snap TRE and fixed T00 lung-mask foreground regularity.
- Sweep alphas: `10000.000000, 100000.000000`.
- The alpha `10000` row is imported from the completed anchor run unless `--no-anchor` is used.

## Completion

- Completed case rows: `20`.
- Completion errors: `[]`.

## Mean Rows

| Alpha | Cases | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0 % | FG BE | FG diffusion | Sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10000.000000 | 10 | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 | 0.006341 | 0.016738 | anchor_alpha10000 |
| 100000.000000 | 10 | 1.003 +/- 0.154 | 0.155840 +/- 0.033335 | 0.0005 | 0.001099 | 0.008170 | sweep |

## Current Reference

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- |
| Current full-crop NCC + dense FD-BE, alpha 10000 | 0.966 +/- 0.151 | 0.185452 +/- 0.060879 | 0.0037 | 0.001971 | 0.010033 |

## Provisional Decision

- Best corrected full point by TRE: alpha `100000.000000` with TRE `1.003` and FG stdLogJ `0.155840`.
- Best completed alpha is the high endpoint `100000`; expand upward or test `/3` and `x3` around it.

## Artifacts

- `case_rows.csv`, `case_rows.json`
- `mean_rows.csv`, `mean_rows.json`
- `comparison_rows.csv`, `comparison_rows.json`
