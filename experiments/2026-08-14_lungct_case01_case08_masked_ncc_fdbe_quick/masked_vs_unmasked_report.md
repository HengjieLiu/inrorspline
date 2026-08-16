# LungCT masked-NCC FD-BE quick A/B

- Updated: `2026-08-14 22:29:17 +0000`
- Comparison: prior full-crop NCC FD-BE runs versus new fixed-T00 lung-mask NCC FD-BE runs.
- Scope: DIR-Lab 4DCT case01 and case08 only; this is a quick protocol check, not a population conclusion.
- Caution: this isolates the NCC data-loss domain inside the FD-BE family; it is not a native-IDIR equivalence claim.

| Case | Method | TRE old | TRE masked | Delta | stdLogJ old | stdLogJ masked | Lung 1-NCC old | Lung 1-NCC masked |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| case01 | INR-BSCP | 0.796 | 0.873 | 0.077 | 0.1089 | 0.2032 | 0.03186 | 0.02178 |
| case01 | D-BSCP | 0.749 | 1.601 | 0.852 | 0.1101 | 0.4522 | 0.02819 | 0.02670 |
| case01 | MR-D-BSCP | 0.755 | 0.915 | 0.160 | 0.1096 | 0.2205 | 0.02814 | 0.01904 |
| case08 | INR-BSCP | 6.886 | 8.297 | 1.410 | 0.4764 | 0.8122 | 0.09147 | 0.05856 |
| case08 | D-BSCP | 11.549 | 14.154 | 2.605 | 0.6154 | 1.0104 | 0.11990 | 0.07679 |
| case08 | MR-D-BSCP | 1.087 | 1.054 | -0.033 | 0.1922 | 0.3518 | 0.03622 | 0.02274 |

Generated files:

- `masked_vs_unmasked_long.csv`
- `masked_vs_unmasked_summary.csv`
- `masked_vs_unmasked_manifest.json`
