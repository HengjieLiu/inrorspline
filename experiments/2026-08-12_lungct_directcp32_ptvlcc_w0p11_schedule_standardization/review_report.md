# LungCT MR-D-BSCP pTVReg-Loss Schedule Standardization

## Validation

- Errors: `[]`.
- New rows: `10`.
- New stage epochs: `400,400,800,900`.

## Aggregate Comparison

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0% mean +/- SD | Elapsed sec | Peak GB |
| --- | --- | --- | --- | --- | --- |
| 2400-step pTVReg-loss baseline w=0.11 | 0.922 +/- 0.152 | 0.179961 +/- 0.049568 | 0.0138 +/- 0.0193 | 234.1 +/- 42.6 | 5.48 +/- 1.07 |
| 2500-step pTVReg-loss pilot w=0.11 | 0.924 +/- 0.152 | 0.179973 +/- 0.049585 | 0.0137 +/- 0.0191 | 155.6 +/- 20.1 | 5.39 +/- 1.05 |

## Case TRE Comparison

| Case | 2400 TRE | 2500 TRE | Delta | New FG stdLogJ | New FG J<=0% | Elapsed sec | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | 0.774 | 0.774 | 0.000 | 0.100638 | 0.0000 | 128.9 | 3.90 |
| 02 | 0.775 | 0.775 | 0.000 | 0.104487 | 0.0000 | 196.2 | 7.33 |
| 03 | 0.923 | 0.928 | 0.005 | 0.119713 | 0.0001 | 160.9 | 6.03 |
| 04 | 1.275 | 1.275 | 0.001 | 0.210599 | 0.0000 | 156.3 | 5.33 |
| 05 | 1.087 | 1.084 | -0.003 | 0.227116 | 0.0392 | 150.5 | 5.61 |
| 06 | 0.836 | 0.843 | 0.007 | 0.236028 | 0.0253 | 151.2 | 5.17 |
| 07 | 0.806 | 0.808 | 0.002 | 0.191851 | 0.0012 | 161.1 | 5.68 |
| 08 | 1.005 | 1.010 | 0.004 | 0.217271 | 0.0561 | 160.4 | 6.38 |
| 09 | 0.910 | 0.913 | 0.003 | 0.178495 | 0.0010 | 119.4 | 3.58 |
| 10 | 0.827 | 0.827 | 0.000 | 0.213534 | 0.0145 | 170.7 | 4.91 |

## Artifacts

- `case_rows.csv`, `case_rows.json`
- `baseline_case_rows.csv`, `baseline_case_rows.json`
- `comparison_rows.csv`, `comparison_rows.json`
