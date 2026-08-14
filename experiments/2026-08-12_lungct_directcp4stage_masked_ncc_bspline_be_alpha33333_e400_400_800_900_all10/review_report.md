# LungCT Corrected MR-D-BSCP IDIR-Loss Schedule Standardization

## Validation

- Errors: `[]`.
- New rows: `10`.
- New stage epochs: `400,400,800,900`.

## Aggregate Comparison

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0% mean +/- SD | Elapsed sec | Peak GB |
| --- | --- | --- | --- | --- | --- |
| 2400-step corrected IDIR-loss alpha=10000 | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 +/- 0.0579 | 785.0 +/- 33.2 | 5.49 +/- 1.07 |
| 2400-step corrected IDIR-loss alpha=100000 | 1.003 +/- 0.154 | 0.155840 +/- 0.033335 | 0.0005 +/- 0.0014 | 682.3 +/- 25.6 | 5.39 +/- 1.05 |
| 2500-step corrected IDIR-loss alpha=33333.333333 | 1.016 +/- 0.160 | 0.195866 +/- 0.042067 | 0.0058 +/- 0.0151 | 699.6 +/- 22.3 | 5.39 +/- 1.05 |

## Case TRE Comparison

| Case | 2400 TRE | 2500 TRE | Delta | New FG stdLogJ | New FG J<=0% | Elapsed sec | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | n/a | 0.813 | n/a | 0.131031 | 0.0000 | 687.5 | 3.90 |
| 02 | n/a | 0.844 | n/a | 0.137042 | 0.0000 | 727.9 | 7.33 |
| 03 | n/a | 0.967 | n/a | 0.148358 | 0.0000 | 681.3 | 6.03 |
| 04 | n/a | 1.372 | n/a | 0.221529 | 0.0000 | 659.1 | 5.33 |
| 05 | n/a | 1.165 | n/a | 0.206278 | 0.0023 | 698.1 | 5.61 |
| 06 | n/a | 1.039 | n/a | 0.262123 | 0.0040 | 728.9 | 5.17 |
| 07 | n/a | 0.898 | n/a | 0.216902 | 0.0001 | 697.6 | 5.68 |
| 08 | n/a | 1.134 | n/a | 0.240601 | 0.0509 | 730.7 | 6.38 |
| 09 | n/a | 0.978 | n/a | 0.188182 | 0.0000 | 699.9 | 3.58 |
| 10 | n/a | 0.955 | n/a | 0.206613 | 0.0005 | 684.9 | 4.91 |

## Artifacts

- `case_rows.csv`, `case_rows.json`
- `baseline_case_rows.csv`, `baseline_case_rows.json`
- `comparison_rows.csv`, `comparison_rows.json`
