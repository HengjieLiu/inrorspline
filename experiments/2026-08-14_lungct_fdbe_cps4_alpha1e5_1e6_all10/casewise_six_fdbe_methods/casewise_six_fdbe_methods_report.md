# LungCT FD-BE cps4 alpha sweep: casewise report

- Updated: `2026-08-15 00:39:42 +0000`
- Scope: INR-BSCP cps4 and D-BSCP cps4, unmasked full-crop global NCC + dense FD-BE.
- Cases: DIR-Lab 4DCT cases 01-10, pTVReg iso1mm landmark-crop protocol.
- Headline metric: original/native snap-to-voxel TRE in mm; regularity uses foreground SDlogJ.
- Fixed T00 lung-mask NCC is a diagnostic/evaluation field only, not the optimization loss domain.

## Validation

- PASS: source has 60 rows.
- PASS: inr_bscp alpha=1e4 has 10/10 cases.
- PASS: inr_bscp alpha=1e5 has 10/10 cases.
- PASS: inr_bscp alpha=1e6 has 10/10 cases.
- PASS: d_bscp alpha=1e4 has 10/10 cases.
- PASS: d_bscp alpha=1e5 has 10/10 cases.
- PASS: d_bscp alpha=1e6 has 10/10 cases.
- PASS: no masked-loss method IDs or masked source paths detected.
- PASS: all rows have finite original_snap TRE and foreground SDlogJ.
- PASS: new alpha=1e5/1e6 rows include full-crop and fixed-mask 1-NCC diagnostics.
- Source: `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/fdbe_cps4_alpha_sweep_case_rows.csv`

## Overall Means

| Method | alpha | Cases | TRE mean | TRE SD | SDlogJ mean | Folding % | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 10 | 1.762 | 1.841 | 0.2360 | 0.2308 | NA | NA | 246s | 3.45 |
| INR-BSCP cps4 (FD-BE) | 1e5 | 10 | 1.043 | 0.171 | 0.1193 | 0.0000 | 0.00529 | 0.05004 | 245s | 3.46 |
| INR-BSCP cps4 (FD-BE) | 1e6 | 10 | 1.218 | 0.280 | 0.0858 | 0.0000 | 0.00817 | 0.07275 | 248s | 3.46 |
| D-BSCP cps4 (FD-BE) | 1e4 | 10 | 3.894 | 3.455 | 0.4265 | 1.4257 | NA | NA | 207s | 2.35 |
| D-BSCP cps4 (FD-BE) | 1e5 | 10 | 1.016 | 0.200 | 0.1182 | 0.0000 | 0.00487 | 0.04621 | 204s | 2.36 |
| D-BSCP cps4 (FD-BE) | 1e6 | 10 | 1.209 | 0.279 | 0.0857 | 0.0000 | 0.00800 | 0.07153 | 203s | 2.36 |

## Best Alpha By Case

| Case | Method | Best alpha | TRE | SDlogJ | Folding % |
| --- | --- | --- | --- | --- | --- |
| 01 | INR-BSCP cps4 (FD-BE) | 1e4 | 0.796 | 0.1089 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 1e5 | 0.739 | 0.0706 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 1e5 | 0.735 | 0.0664 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 1e5 | 0.732 | 0.0656 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 1e5 | 0.840 | 0.0822 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | 1e5 | 0.853 | 0.0829 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 1e5 | 1.227 | 0.1617 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 1e5 | 1.258 | 0.1612 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | 1e5 | 1.326 | 0.1124 | 0.0000 |
| 05 | D-BSCP cps4 (FD-BE) | 1e5 | 1.336 | 0.1128 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | 1e5 | 1.008 | 0.1654 | 0.0000 |
| 06 | D-BSCP cps4 (FD-BE) | 1e5 | 0.991 | 0.1656 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | 1e5 | 1.090 | 0.1344 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | 1e5 | 1.085 | 0.1339 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | 1e5 | 1.127 | 0.1296 | 0.0000 |
| 08 | D-BSCP cps4 (FD-BE) | 1e5 | 1.105 | 0.1287 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 1e4 | 0.956 | 0.1911 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | 1e4 | 0.972 | 0.1943 | 0.0070 |
| 10 | INR-BSCP cps4 (FD-BE) | 1e4 | 0.968 | 0.1957 | 0.0092 |
| 10 | D-BSCP cps4 (FD-BE) | 1e5 | 1.045 | 0.1362 | 0.0000 |

## Delta Versus alpha=1e4

| Case | Method | alpha | Delta TRE | Delta SDlogJ | Delta folding % |
| --- | --- | --- | --- | --- | --- |
| 01 | INR-BSCP cps4 (FD-BE) | 1e5 | 0.202 | -0.0285 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | 1e6 | 0.037 | -0.0564 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 1e5 | -0.010 | -0.0394 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 1e6 | 0.069 | -0.0575 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.030 | -0.0389 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 1e6 | 0.065 | -0.0555 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 1e5 | -0.157 | -0.1532 | -0.2314 |
| 02 | D-BSCP cps4 (FD-BE) | 1e6 | -0.054 | -0.1692 | -0.2314 |
| 03 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.095 | -0.0415 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 1e6 | 0.042 | -0.0633 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | 1e5 | -0.748 | -0.2120 | -0.4971 |
| 03 | D-BSCP cps4 (FD-BE) | 1e6 | -0.617 | -0.2348 | -0.4971 |
| 04 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.079 | -0.0679 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 1e6 | -0.033 | -0.1080 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 1e5 | -1.035 | -0.2555 | -1.1387 |
| 04 | D-BSCP cps4 (FD-BE) | 1e6 | -1.007 | -0.2955 | -1.1387 |
| 05 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.536 | -0.1447 | -0.2756 |
| 05 | INR-BSCP cps4 (FD-BE) | 1e6 | -0.129 | -0.1853 | -0.2756 |
| 05 | D-BSCP cps4 (FD-BE) | 1e5 | -2.485 | -0.2776 | -0.5859 |
| 05 | D-BSCP cps4 (FD-BE) | 1e6 | -2.091 | -0.3185 | -0.5859 |
| 06 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.367 | -0.2188 | -0.6834 |
| 06 | INR-BSCP cps4 (FD-BE) | 1e6 | -0.189 | -0.2689 | -0.6834 |
| 06 | D-BSCP cps4 (FD-BE) | 1e5 | -5.381 | -0.6955 | -5.1062 |
| 06 | D-BSCP cps4 (FD-BE) | 1e6 | -5.232 | -0.7458 | -5.1062 |
| 07 | INR-BSCP cps4 (FD-BE) | 1e5 | -0.679 | -0.1535 | -0.2438 |
| 07 | INR-BSCP cps4 (FD-BE) | 1e6 | -0.339 | -0.1847 | -0.2438 |
| 07 | D-BSCP cps4 (FD-BE) | 1e5 | -5.527 | -0.5259 | -2.6259 |
| 07 | D-BSCP cps4 (FD-BE) | 1e6 | -5.191 | -0.5568 | -2.6259 |
| 08 | INR-BSCP cps4 (FD-BE) | 1e5 | -5.759 | -0.3468 | -1.0963 |
| 08 | INR-BSCP cps4 (FD-BE) | 1e6 | -5.490 | -0.3776 | -1.0963 |
| 08 | D-BSCP cps4 (FD-BE) | 1e5 | -10.444 | -0.4867 | -2.6500 |
| 08 | D-BSCP cps4 (FD-BE) | 1e6 | -10.176 | -0.5166 | -2.6500 |
| 09 | INR-BSCP cps4 (FD-BE) | 1e5 | 0.068 | -0.0667 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 1e6 | 0.302 | -0.1013 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | 1e5 | 0.041 | -0.0701 | -0.0070 |
| 09 | D-BSCP cps4 (FD-BE) | 1e6 | 0.256 | -0.1046 | -0.0070 |
| 10 | INR-BSCP cps4 (FD-BE) | 1e5 | 0.090 | -0.0594 | -0.0092 |
| 10 | INR-BSCP cps4 (FD-BE) | 1e6 | 0.298 | -0.1006 | -0.0092 |
| 10 | D-BSCP cps4 (FD-BE) | 1e5 | -3.034 | -0.3676 | -1.4149 |
| 10 | D-BSCP cps4 (FD-BE) | 1e6 | -2.810 | -0.4088 | -1.4149 |

## Case Tables

### Case 01

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 0.796 | 0.1089 | 0.0000 | 0.0005 | 0.0036 | NA | NA | 177s | 2.51 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 0.999 | 0.0803 | 0.0000 | 0.0003 | 0.0025 | 0.00408 | 0.05243 | 176s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 0.833 | 0.0525 | 0.0000 | 0.0000 | 0.0014 | 0.00403 | 0.05252 | 176s | 2.52 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 0.749 | 0.1101 | 0.0000 | 0.0007 | 0.0035 | NA | NA | 146s | 1.70 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 0.739 | 0.0706 | 0.0000 | 0.0001 | 0.0021 | 0.00260 | 0.03742 | 147s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 0.818 | 0.0526 | 0.0000 | 0.0000 | 0.0015 | 0.00396 | 0.05231 | 145s | 1.70 | completed |

### Case 02

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 0.765 | 0.1052 | 0.0000 | 0.0006 | 0.0053 | NA | NA | 321s | 4.63 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 0.735 | 0.0664 | 0.0000 | 0.0001 | 0.0032 | 0.00220 | 0.02999 | 318s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 0.829 | 0.0497 | 0.0000 | 0.0000 | 0.0022 | 0.00361 | 0.04613 | 317s | 4.65 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 0.889 | 0.2188 | 0.2314 | 0.0014 | 0.0089 | NA | NA | 280s | 3.19 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 0.732 | 0.0656 | 0.0000 | 0.0001 | 0.0031 | 0.00203 | 0.02800 | 265s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 0.835 | 0.0496 | 0.0000 | 0.0000 | 0.0022 | 0.00351 | 0.04515 | 264s | 3.20 | completed |

### Case 03

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 0.936 | 0.1237 | 0.0000 | 0.0007 | 0.0053 | NA | NA | 273s | 3.85 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 0.840 | 0.0822 | 0.0000 | 0.0001 | 0.0035 | 0.00255 | 0.02864 | 277s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 0.978 | 0.0604 | 0.0000 | 0.0000 | 0.0026 | 0.00412 | 0.04363 | 282s | 3.86 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 1.601 | 0.2949 | 0.4971 | 0.0025 | 0.0154 | NA | NA | 228s | 2.63 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 0.853 | 0.0829 | 0.0000 | 0.0002 | 0.0035 | 0.00240 | 0.02716 | 215s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 0.984 | 0.0601 | 0.0000 | 0.0000 | 0.0027 | 0.00397 | 0.04202 | 219s | 2.64 | completed |

### Case 04

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 1.306 | 0.2296 | 0.0000 | 0.0019 | 0.0104 | NA | NA | 251s | 3.39 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.227 | 0.1617 | 0.0000 | 0.0004 | 0.0062 | 0.00580 | 0.03932 | 244s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.273 | 0.1216 | 0.0000 | 0.0001 | 0.0042 | 0.00921 | 0.05847 | 249s | 3.40 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 2.293 | 0.4167 | 1.1387 | 0.0055 | 0.0284 | NA | NA | 216s | 2.32 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.258 | 0.1612 | 0.0000 | 0.0004 | 0.0063 | 0.00533 | 0.03617 | 196s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.286 | 0.1212 | 0.0000 | 0.0001 | 0.0042 | 0.00902 | 0.05768 | 198s | 2.33 | completed |

### Case 05

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 1.862 | 0.2571 | 0.2756 | 0.0017 | 0.0124 | NA | NA | 260s | 3.59 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.326 | 0.1124 | 0.0000 | 0.0003 | 0.0063 | 0.00318 | 0.04871 | 256s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.733 | 0.0718 | 0.0000 | 0.0001 | 0.0040 | 0.00632 | 0.08549 | 264s | 3.59 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 3.822 | 0.3904 | 0.5859 | 0.0033 | 0.0217 | NA | NA | 221s | 2.44 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.336 | 0.1128 | 0.0000 | 0.0003 | 0.0063 | 0.00291 | 0.04550 | 216s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.731 | 0.0719 | 0.0000 | 0.0001 | 0.0041 | 0.00616 | 0.08426 | 211s | 2.45 | completed |

### Case 06

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 1.375 | 0.3842 | 0.6834 | 0.0052 | 0.0236 | NA | NA | 235s | 3.33 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.008 | 0.1654 | 0.0000 | 0.0005 | 0.0087 | 0.01231 | 0.08761 | 230s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.186 | 0.1153 | 0.0000 | 0.0001 | 0.0062 | 0.01719 | 0.11398 | 235s | 3.34 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 6.373 | 0.8610 | 5.1062 | 0.0172 | 0.0793 | NA | NA | 200s | 2.26 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 0.991 | 0.1656 | 0.0000 | 0.0006 | 0.0088 | 0.01183 | 0.08507 | 208s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.141 | 0.1152 | 0.0000 | 0.0001 | 0.0062 | 0.01695 | 0.11201 | 196s | 2.26 | completed |

### Case 07

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 1.768 | 0.2879 | 0.2438 | 0.0027 | 0.0175 | NA | NA | 249s | 3.64 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.090 | 0.1344 | 0.0000 | 0.0003 | 0.0079 | 0.00614 | 0.05361 | 252s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.429 | 0.1033 | 0.0000 | 0.0001 | 0.0063 | 0.00944 | 0.07818 | 260s | 3.65 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 6.612 | 0.6598 | 2.6259 | 0.0112 | 0.0553 | NA | NA | 214s | 2.48 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.085 | 0.1339 | 0.0000 | 0.0004 | 0.0080 | 0.00581 | 0.05146 | 218s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.421 | 0.1030 | 0.0000 | 0.0001 | 0.0063 | 0.00929 | 0.07742 | 218s | 2.49 | completed |

### Case 08

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 6.886 | 0.4764 | 1.0963 | 0.0043 | 0.0378 | NA | NA | 286s | 4.06 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.127 | 0.1296 | 0.0000 | 0.0003 | 0.0103 | 0.00457 | 0.05664 | 288s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.397 | 0.0988 | 0.0000 | 0.0001 | 0.0085 | 0.00830 | 0.09140 | 293s | 4.07 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 11.549 | 0.6154 | 2.6500 | 0.0082 | 0.0536 | NA | NA | 233s | 2.78 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.105 | 0.1287 | 0.0000 | 0.0003 | 0.0103 | 0.00424 | 0.05271 | 236s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.373 | 0.0988 | 0.0000 | 0.0001 | 0.0085 | 0.00795 | 0.08893 | 252s | 2.79 | completed |

### Case 09

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 0.956 | 0.1911 | 0.0000 | 0.0018 | 0.0092 | NA | NA | 167s | 2.31 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.024 | 0.1244 | 0.0000 | 0.0003 | 0.0054 | 0.00712 | 0.05273 | 173s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.258 | 0.0898 | 0.0000 | 0.0000 | 0.0036 | 0.01086 | 0.07642 | 179s | 2.32 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 0.972 | 0.1943 | 0.0070 | 0.0023 | 0.0095 | NA | NA | 149s | 1.56 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.013 | 0.1242 | 0.0000 | 0.0003 | 0.0054 | 0.00682 | 0.05081 | 137s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.227 | 0.0898 | 0.0000 | 0.0000 | 0.0037 | 0.01069 | 0.07538 | 133s | 1.56 | completed |

### Case 10

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | 1e4 | 0.968 | 0.1957 | 0.0092 | 0.0015 | 0.0127 | NA | NA | 239s | 3.16 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 1e5 | 1.057 | 0.1363 | 0.0000 | 0.0003 | 0.0088 | 0.00500 | 0.05075 | 239s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | 1e6 | 1.265 | 0.0951 | 0.0000 | 0.0001 | 0.0063 | 0.00864 | 0.08130 | 226s | 3.17 | completed |
| D-BSCP cps4 (FD-BE) | 1e4 | 4.079 | 0.5038 | 1.4149 | 0.0060 | 0.0363 | NA | NA | 182s | 2.14 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 1e5 | 1.045 | 0.1362 | 0.0000 | 0.0003 | 0.0089 | 0.00468 | 0.04777 | 197s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | 1e6 | 1.270 | 0.0950 | 0.0000 | 0.0001 | 0.0063 | 0.00850 | 0.08011 | 194s | 2.15 | completed |

## Companion CSVs

- `casewise_long.csv`: cleaned source rows for the six method/alpha combinations.
- `casewise_tre_wide.csv`: case-by-case original_snap TRE matrix.
- `casewise_stdlogj_wide.csv`: case-by-case foreground SDlogJ matrix.
- `casewise_best_by_case_method.csv`: lowest-TRE alpha per case and method.
- `casewise_delta_vs_alpha10000.csv`: per-case deltas for alpha=1e5/1e6 versus alpha=1e4.
