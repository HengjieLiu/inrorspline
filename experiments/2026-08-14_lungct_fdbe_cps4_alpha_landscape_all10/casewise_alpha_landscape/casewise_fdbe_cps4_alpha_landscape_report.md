# LungCT FD-BE cps4 alpha sweep: casewise report

- Updated: `2026-08-15 03:36:59 +0000`
- Scope: INR-BSCP cps4 and D-BSCP cps4, unmasked full-crop global NCC + dense FD-BE.
- Cases: DIR-Lab 4DCT cases 01-10, pTVReg iso1mm landmark-crop protocol.
- Headline metric: original/native snap-to-voxel TRE in mm; regularity uses foreground SDlogJ.
- Fixed T00 lung-mask NCC is a diagnostic/evaluation field only, not the optimization loss domain.

## Validation

- PASS: source has 140 rows.
- PASS: inr_bscp alpha=e4 has 10/10 cases.
- PASS: inr_bscp alpha=2e4 has 10/10 cases.
- PASS: inr_bscp alpha=5e4 has 10/10 cases.
- PASS: inr_bscp alpha=e5 has 10/10 cases.
- PASS: inr_bscp alpha=2e5 has 10/10 cases.
- PASS: inr_bscp alpha=5e5 has 10/10 cases.
- PASS: inr_bscp alpha=e6 has 10/10 cases.
- PASS: d_bscp alpha=e4 has 10/10 cases.
- PASS: d_bscp alpha=2e4 has 10/10 cases.
- PASS: d_bscp alpha=5e4 has 10/10 cases.
- PASS: d_bscp alpha=e5 has 10/10 cases.
- PASS: d_bscp alpha=2e5 has 10/10 cases.
- PASS: d_bscp alpha=5e5 has 10/10 cases.
- PASS: d_bscp alpha=e6 has 10/10 cases.
- PASS: no masked-loss method IDs or masked source paths detected.
- PASS: all rows have finite original_snap TRE and foreground SDlogJ.
- PASS: new alpha=1e5/1e6 rows include full-crop and fixed-mask 1-NCC diagnostics.
- Source: `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/fdbe_cps4_alpha_sweep_case_rows.csv`

## Overall Means

| Method | alpha | Cases | TRE mean | TRE SD | SDlogJ mean | Folding % | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 10 | 1.762 | 1.841 | 0.2360 | 0.2308 | 0.00415 | 0.04304 | 242s | 3.46 |
| INR-BSCP cps4 (FD-BE) | 2e4 | 10 | 1.474 | 1.600 | 0.1795 | 0.0659 | 0.00431 | 0.04402 | 243s | 3.46 |
| INR-BSCP cps4 (FD-BE) | 5e4 | 10 | 1.042 | 0.221 | 0.1351 | 0.0000 | 0.00470 | 0.04496 | 240s | 3.46 |
| INR-BSCP cps4 (FD-BE) | e5 | 10 | 1.043 | 0.171 | 0.1193 | 0.0000 | 0.00529 | 0.05004 | 245s | 3.46 |
| INR-BSCP cps4 (FD-BE) | 2e5 | 10 | 1.069 | 0.228 | 0.1068 | 0.0000 | 0.00582 | 0.05385 | 247s | 3.46 |
| INR-BSCP cps4 (FD-BE) | 5e5 | 10 | 1.156 | 0.269 | 0.0940 | 0.0000 | 0.00705 | 0.06406 | 243s | 3.46 |
| INR-BSCP cps4 (FD-BE) | e6 | 10 | 1.218 | 0.280 | 0.0858 | 0.0000 | 0.00817 | 0.07275 | 248s | 3.46 |
| D-BSCP cps4 (FD-BE) | e4 | 10 | 3.894 | 3.455 | 0.4265 | 1.4257 | 0.00528 | 0.05882 | 203s | 2.36 |
| D-BSCP cps4 (FD-BE) | 2e4 | 10 | 2.743 | 3.019 | 0.2854 | 0.4920 | 0.00491 | 0.05324 | 203s | 2.36 |
| D-BSCP cps4 (FD-BE) | 5e4 | 10 | 1.306 | 0.790 | 0.1592 | 0.0749 | 0.00458 | 0.04643 | 199s | 2.36 |
| D-BSCP cps4 (FD-BE) | e5 | 10 | 1.016 | 0.200 | 0.1182 | 0.0000 | 0.00487 | 0.04621 | 204s | 2.36 |
| D-BSCP cps4 (FD-BE) | 2e5 | 10 | 1.057 | 0.230 | 0.1065 | 0.0000 | 0.00558 | 0.05212 | 208s | 2.36 |
| D-BSCP cps4 (FD-BE) | 5e5 | 10 | 1.144 | 0.262 | 0.0940 | 0.0000 | 0.00681 | 0.06204 | 203s | 2.36 |
| D-BSCP cps4 (FD-BE) | e6 | 10 | 1.209 | 0.279 | 0.0857 | 0.0000 | 0.00800 | 0.07153 | 203s | 2.36 |

## Best Alpha By Case

| Case | Method | Best alpha | TRE | SDlogJ | Folding % |
| --- | --- | --- | --- | --- | --- |
| 01 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.724 | 0.0937 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 5e4 | 0.724 | 0.0787 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.712 | 0.0595 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 2e5 | 0.709 | 0.0590 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | e5 | 0.840 | 0.0822 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | e5 | 0.853 | 0.0829 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | e5 | 1.227 | 0.1617 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 5e4 | 1.253 | 0.1769 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | 2e4 | 1.146 | 0.1518 | 0.0000 |
| 05 | D-BSCP cps4 (FD-BE) | e5 | 1.336 | 0.1128 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | 2e4 | 1.006 | 0.2545 | 0.0246 |
| 06 | D-BSCP cps4 (FD-BE) | e5 | 0.991 | 0.1656 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.928 | 0.1784 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | 5e4 | 1.024 | 0.1498 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | e5 | 1.127 | 0.1296 | 0.0000 |
| 08 | D-BSCP cps4 (FD-BE) | e5 | 1.105 | 0.1287 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | e4 | 0.956 | 0.1911 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | e4 | 0.972 | 0.1943 | 0.0070 |
| 10 | INR-BSCP cps4 (FD-BE) | e4 | 0.968 | 0.1957 | 0.0092 |
| 10 | D-BSCP cps4 (FD-BE) | 5e4 | 0.989 | 0.1494 | 0.0000 |

## Delta Versus alpha=e5

| Case | Method | alpha | Delta TRE | Delta SDlogJ | Delta folding % |
| --- | --- | --- | --- | --- | --- |
| 01 | INR-BSCP cps4 (FD-BE) | e4 | -0.202 | 0.0285 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | 2e4 | -0.275 | 0.0134 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | 5e4 | -0.274 | -0.0016 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | 2e5 | -0.226 | -0.0162 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | 5e5 | -0.206 | -0.0231 | 0.0000 |
| 01 | INR-BSCP cps4 (FD-BE) | e6 | -0.165 | -0.0278 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | e4 | 0.010 | 0.0394 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 2e4 | 0.004 | 0.0233 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 5e4 | -0.015 | 0.0081 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 2e5 | 0.006 | -0.0067 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | 5e5 | 0.066 | -0.0132 | 0.0000 |
| 01 | D-BSCP cps4 (FD-BE) | e6 | 0.079 | -0.0180 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | e4 | 0.030 | 0.0389 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.027 | 0.0243 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.033 | 0.0160 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 2e5 | -0.023 | -0.0069 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.053 | -0.0129 | 0.0000 |
| 02 | INR-BSCP cps4 (FD-BE) | e6 | 0.094 | -0.0167 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | e4 | 0.157 | 0.1532 | 0.2314 |
| 02 | D-BSCP cps4 (FD-BE) | 2e4 | 0.008 | 0.0238 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 5e4 | -0.014 | 0.0087 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 2e5 | -0.023 | -0.0066 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | 5e5 | 0.046 | -0.0125 | 0.0000 |
| 02 | D-BSCP cps4 (FD-BE) | e6 | 0.104 | -0.0160 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | e4 | 0.095 | 0.0415 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.061 | 0.0262 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.029 | 0.0096 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.044 | -0.0070 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.083 | -0.0160 | 0.0000 |
| 03 | INR-BSCP cps4 (FD-BE) | e6 | 0.138 | -0.0218 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | e4 | 0.748 | 0.2120 | 0.4971 |
| 03 | D-BSCP cps4 (FD-BE) | 2e4 | 0.037 | 0.0268 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | 5e4 | 0.024 | 0.0095 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | 2e5 | 0.054 | -0.0079 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | 5e5 | 0.075 | -0.0168 | 0.0000 |
| 03 | D-BSCP cps4 (FD-BE) | e6 | 0.131 | -0.0227 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | e4 | 0.079 | 0.0679 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.017 | 0.0416 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.016 | 0.0155 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.068 | -0.0138 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.110 | -0.0291 | 0.0000 |
| 04 | INR-BSCP cps4 (FD-BE) | e6 | 0.046 | -0.0401 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | e4 | 1.035 | 0.2555 | 1.1387 |
| 04 | D-BSCP cps4 (FD-BE) | 2e4 | 0.015 | 0.0442 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 5e4 | -0.005 | 0.0157 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 2e5 | 0.016 | -0.0135 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | 5e5 | 0.043 | -0.0289 | 0.0000 |
| 04 | D-BSCP cps4 (FD-BE) | e6 | 0.028 | -0.0399 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | e4 | 0.536 | 0.1447 | 0.2756 |
| 05 | INR-BSCP cps4 (FD-BE) | 2e4 | -0.180 | 0.0394 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | 5e4 | -0.002 | 0.0156 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.117 | -0.0129 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.326 | -0.0297 | 0.0000 |
| 05 | INR-BSCP cps4 (FD-BE) | e6 | 0.407 | -0.0406 | 0.0000 |
| 05 | D-BSCP cps4 (FD-BE) | e4 | 2.485 | 0.2776 | 0.5859 |
| 05 | D-BSCP cps4 (FD-BE) | 2e4 | 2.135 | 0.2621 | 0.5249 |
| 05 | D-BSCP cps4 (FD-BE) | 5e4 | 0.932 | 0.1047 | 0.2625 |
| 05 | D-BSCP cps4 (FD-BE) | 2e5 | 0.108 | -0.0141 | 0.0000 |
| 05 | D-BSCP cps4 (FD-BE) | 5e5 | 0.307 | -0.0299 | 0.0000 |
| 05 | D-BSCP cps4 (FD-BE) | e6 | 0.394 | -0.0409 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | e4 | 0.367 | 0.2188 | 0.6834 |
| 06 | INR-BSCP cps4 (FD-BE) | 2e4 | -0.003 | 0.0892 | 0.0246 |
| 06 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.009 | 0.0281 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.046 | -0.0204 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.123 | -0.0391 | 0.0000 |
| 06 | INR-BSCP cps4 (FD-BE) | e6 | 0.178 | -0.0501 | 0.0000 |
| 06 | D-BSCP cps4 (FD-BE) | e4 | 5.381 | 0.6955 | 5.1062 |
| 06 | D-BSCP cps4 (FD-BE) | 2e4 | 1.688 | 0.3277 | 0.9261 |
| 06 | D-BSCP cps4 (FD-BE) | 5e4 | 0.007 | 0.0298 | 0.0000 |
| 06 | D-BSCP cps4 (FD-BE) | 2e5 | 0.013 | -0.0201 | 0.0000 |
| 06 | D-BSCP cps4 (FD-BE) | 5e5 | 0.105 | -0.0390 | 0.0000 |
| 06 | D-BSCP cps4 (FD-BE) | e6 | 0.150 | -0.0503 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | e4 | 0.679 | 0.1535 | 0.2438 |
| 07 | INR-BSCP cps4 (FD-BE) | 2e4 | -0.162 | 0.0439 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | 5e4 | -0.049 | 0.0171 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.087 | -0.0115 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.197 | -0.0244 | 0.0000 |
| 07 | INR-BSCP cps4 (FD-BE) | e6 | 0.340 | -0.0312 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | e4 | 5.527 | 0.5259 | 2.6259 |
| 07 | D-BSCP cps4 (FD-BE) | 2e4 | 4.181 | 0.4272 | 1.8127 |
| 07 | D-BSCP cps4 (FD-BE) | 5e4 | -0.060 | 0.0158 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | 2e5 | 0.085 | -0.0120 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | 5e5 | 0.184 | -0.0240 | 0.0000 |
| 07 | D-BSCP cps4 (FD-BE) | e6 | 0.336 | -0.0309 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | e4 | 5.759 | 0.3468 | 1.0963 |
| 08 | INR-BSCP cps4 (FD-BE) | 2e4 | 4.879 | 0.2495 | 0.6340 |
| 08 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.250 | 0.0301 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.060 | -0.0125 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.208 | -0.0232 | 0.0000 |
| 08 | INR-BSCP cps4 (FD-BE) | e6 | 0.270 | -0.0308 | 0.0000 |
| 08 | D-BSCP cps4 (FD-BE) | e4 | 10.444 | 0.4867 | 2.6500 |
| 08 | D-BSCP cps4 (FD-BE) | 2e4 | 9.118 | 0.3977 | 1.4950 |
| 08 | D-BSCP cps4 (FD-BE) | 5e4 | 2.064 | 0.1905 | 0.4866 |
| 08 | D-BSCP cps4 (FD-BE) | 2e5 | 0.080 | -0.0116 | 0.0000 |
| 08 | D-BSCP cps4 (FD-BE) | 5e5 | 0.196 | -0.0228 | 0.0000 |
| 08 | D-BSCP cps4 (FD-BE) | e6 | 0.268 | -0.0299 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | e4 | -0.068 | 0.0667 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 2e4 | 0.010 | 0.0395 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 5e4 | 0.029 | 0.0142 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.025 | -0.0127 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.099 | -0.0264 | 0.0000 |
| 09 | INR-BSCP cps4 (FD-BE) | e6 | 0.234 | -0.0346 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | e4 | -0.041 | 0.0701 | 0.0070 |
| 09 | D-BSCP cps4 (FD-BE) | 2e4 | -0.010 | 0.0403 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | 5e4 | 0.025 | 0.0145 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | 2e5 | 0.007 | -0.0122 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | 5e5 | 0.104 | -0.0254 | 0.0000 |
| 09 | D-BSCP cps4 (FD-BE) | e6 | 0.215 | -0.0344 | 0.0000 |
| 10 | INR-BSCP cps4 (FD-BE) | e4 | -0.090 | 0.0594 | 0.0092 |
| 10 | INR-BSCP cps4 (FD-BE) | 2e4 | -0.066 | 0.0352 | 0.0000 |
| 10 | INR-BSCP cps4 (FD-BE) | 5e4 | -0.055 | 0.0134 | 0.0000 |
| 10 | INR-BSCP cps4 (FD-BE) | 2e5 | 0.059 | -0.0116 | 0.0000 |
| 10 | INR-BSCP cps4 (FD-BE) | 5e5 | 0.137 | -0.0292 | 0.0000 |
| 10 | INR-BSCP cps4 (FD-BE) | e6 | 0.208 | -0.0412 | 0.0000 |
| 10 | D-BSCP cps4 (FD-BE) | e4 | 3.034 | 0.3676 | 1.4149 |
| 10 | D-BSCP cps4 (FD-BE) | 2e4 | 0.098 | 0.0990 | 0.1615 |
| 10 | D-BSCP cps4 (FD-BE) | 5e4 | -0.056 | 0.0132 | 0.0000 |
| 10 | D-BSCP cps4 (FD-BE) | 2e5 | 0.069 | -0.0120 | 0.0000 |
| 10 | D-BSCP cps4 (FD-BE) | 5e5 | 0.152 | -0.0287 | 0.0000 |
| 10 | D-BSCP cps4 (FD-BE) | e6 | 0.225 | -0.0413 | 0.0000 |

## Case Tables

### Case 01

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 0.796 | 0.1089 | 0.0000 | 0.0005 | 0.0036 | 0.00224 | 0.03186 | 176s | 2.52 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 0.724 | 0.0937 | 0.0000 | 0.0003 | 0.0030 | 0.00218 | 0.03240 | 179s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 0.725 | 0.0787 | 0.0000 | 0.0002 | 0.0024 | 0.00244 | 0.03563 | 179s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 0.999 | 0.0803 | 0.0000 | 0.0003 | 0.0025 | 0.00408 | 0.05243 | 176s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 0.773 | 0.0642 | 0.0000 | 0.0001 | 0.0019 | 0.00306 | 0.04257 | 180s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 0.793 | 0.0572 | 0.0000 | 0.0000 | 0.0016 | 0.00361 | 0.04819 | 182s | 2.52 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 0.833 | 0.0525 | 0.0000 | 0.0000 | 0.0014 | 0.00403 | 0.05252 | 176s | 2.52 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 0.749 | 0.1101 | 0.0000 | 0.0007 | 0.0035 | 0.00184 | 0.02819 | 142s | 1.70 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 0.743 | 0.0940 | 0.0000 | 0.0004 | 0.0029 | 0.00203 | 0.03061 | 161s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 0.724 | 0.0787 | 0.0000 | 0.0002 | 0.0024 | 0.00233 | 0.03418 | 147s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 0.739 | 0.0706 | 0.0000 | 0.0001 | 0.0021 | 0.00260 | 0.03742 | 147s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 0.745 | 0.0640 | 0.0000 | 0.0001 | 0.0019 | 0.00293 | 0.04112 | 149s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 0.805 | 0.0574 | 0.0000 | 0.0000 | 0.0016 | 0.00346 | 0.04704 | 151s | 1.70 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 0.818 | 0.0526 | 0.0000 | 0.0000 | 0.0015 | 0.00396 | 0.05231 | 145s | 1.70 | completed |

### Case 02

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 0.765 | 0.1052 | 0.0000 | 0.0006 | 0.0053 | 0.00156 | 0.02278 | 322s | 4.65 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 0.762 | 0.0907 | 0.0000 | 0.0004 | 0.0045 | 0.00171 | 0.02437 | 325s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 0.768 | 0.0823 | 0.0000 | 0.0003 | 0.0041 | 0.00227 | 0.03153 | 317s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 0.735 | 0.0664 | 0.0000 | 0.0001 | 0.0032 | 0.00220 | 0.02999 | 318s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 0.712 | 0.0595 | 0.0000 | 0.0001 | 0.0028 | 0.00251 | 0.03342 | 321s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 0.788 | 0.0535 | 0.0000 | 0.0000 | 0.0025 | 0.00310 | 0.04015 | 326s | 4.65 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 0.829 | 0.0497 | 0.0000 | 0.0000 | 0.0022 | 0.00361 | 0.04613 | 317s | 4.65 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 0.889 | 0.2188 | 0.2314 | 0.0014 | 0.0089 | 0.00162 | 0.02497 | 267s | 3.20 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 0.740 | 0.0894 | 0.0000 | 0.0004 | 0.0042 | 0.00151 | 0.02193 | 270s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 0.717 | 0.0744 | 0.0000 | 0.0002 | 0.0035 | 0.00177 | 0.02500 | 265s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 0.732 | 0.0656 | 0.0000 | 0.0001 | 0.0031 | 0.00203 | 0.02800 | 265s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 0.709 | 0.0590 | 0.0000 | 0.0001 | 0.0028 | 0.00234 | 0.03184 | 274s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 0.777 | 0.0531 | 0.0000 | 0.0000 | 0.0025 | 0.00295 | 0.03878 | 265s | 3.20 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 0.835 | 0.0496 | 0.0000 | 0.0000 | 0.0022 | 0.00351 | 0.04515 | 264s | 3.20 | completed |

### Case 03

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 0.936 | 0.1237 | 0.0000 | 0.0007 | 0.0053 | 0.00186 | 0.02183 | 267s | 3.86 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 0.901 | 0.1084 | 0.0000 | 0.0004 | 0.0045 | 0.00201 | 0.02323 | 271s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 0.869 | 0.0918 | 0.0000 | 0.0002 | 0.0039 | 0.00228 | 0.02608 | 264s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 0.840 | 0.0822 | 0.0000 | 0.0001 | 0.0035 | 0.00255 | 0.02864 | 277s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 0.884 | 0.0752 | 0.0000 | 0.0001 | 0.0032 | 0.00298 | 0.03244 | 276s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 0.923 | 0.0662 | 0.0000 | 0.0000 | 0.0029 | 0.00348 | 0.03768 | 266s | 3.86 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 0.978 | 0.0604 | 0.0000 | 0.0000 | 0.0026 | 0.00412 | 0.04363 | 282s | 3.86 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 1.601 | 0.2949 | 0.4971 | 0.0025 | 0.0154 | 0.00228 | 0.02842 | 224s | 2.64 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 0.890 | 0.1097 | 0.0000 | 0.0006 | 0.0046 | 0.00179 | 0.02105 | 225s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 0.876 | 0.0924 | 0.0000 | 0.0003 | 0.0039 | 0.00210 | 0.02421 | 221s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 0.853 | 0.0829 | 0.0000 | 0.0002 | 0.0035 | 0.00240 | 0.02716 | 215s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 0.907 | 0.0750 | 0.0000 | 0.0001 | 0.0032 | 0.00274 | 0.03051 | 230s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 0.928 | 0.0661 | 0.0000 | 0.0000 | 0.0029 | 0.00337 | 0.03646 | 227s | 2.64 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 0.984 | 0.0601 | 0.0000 | 0.0000 | 0.0027 | 0.00397 | 0.04202 | 219s | 2.64 | completed |

### Case 04

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 1.306 | 0.2296 | 0.0000 | 0.0019 | 0.0104 | 0.00381 | 0.02701 | 248s | 3.40 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 1.244 | 0.2033 | 0.0000 | 0.0012 | 0.0087 | 0.00422 | 0.02961 | 251s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.243 | 0.1772 | 0.0000 | 0.0006 | 0.0071 | 0.00491 | 0.03362 | 247s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.227 | 0.1617 | 0.0000 | 0.0004 | 0.0062 | 0.00580 | 0.03932 | 244s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.295 | 0.1479 | 0.0000 | 0.0002 | 0.0054 | 0.00645 | 0.04286 | 259s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.337 | 0.1326 | 0.0000 | 0.0001 | 0.0047 | 0.00814 | 0.05268 | 253s | 3.40 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.273 | 0.1216 | 0.0000 | 0.0001 | 0.0042 | 0.00921 | 0.05847 | 249s | 3.40 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 2.293 | 0.4167 | 1.1387 | 0.0055 | 0.0284 | 0.00465 | 0.03524 | 196s | 2.33 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 1.273 | 0.2053 | 0.0000 | 0.0014 | 0.0095 | 0.00388 | 0.02749 | 202s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 1.253 | 0.1769 | 0.0000 | 0.0007 | 0.0073 | 0.00462 | 0.03198 | 194s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.258 | 0.1612 | 0.0000 | 0.0004 | 0.0063 | 0.00533 | 0.03617 | 196s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.274 | 0.1477 | 0.0000 | 0.0002 | 0.0055 | 0.00622 | 0.04137 | 203s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.302 | 0.1322 | 0.0000 | 0.0001 | 0.0047 | 0.00772 | 0.04995 | 200s | 2.33 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.286 | 0.1212 | 0.0000 | 0.0001 | 0.0042 | 0.00902 | 0.05768 | 198s | 2.33 | completed |

### Case 05

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 1.862 | 0.2571 | 0.2756 | 0.0017 | 0.0124 | 0.00234 | 0.03937 | 251s | 3.59 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 1.146 | 0.1518 | 0.0000 | 0.0007 | 0.0077 | 0.00227 | 0.03653 | 248s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.324 | 0.1280 | 0.0000 | 0.0005 | 0.0070 | 0.00294 | 0.04595 | 249s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.326 | 0.1124 | 0.0000 | 0.0003 | 0.0063 | 0.00318 | 0.04871 | 256s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.443 | 0.0995 | 0.0000 | 0.0002 | 0.0058 | 0.00373 | 0.05608 | 256s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.653 | 0.0827 | 0.0000 | 0.0001 | 0.0048 | 0.00500 | 0.07165 | 249s | 3.59 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.733 | 0.0718 | 0.0000 | 0.0001 | 0.0040 | 0.00632 | 0.08549 | 264s | 3.59 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 3.822 | 0.3904 | 0.5859 | 0.0033 | 0.0217 | 0.00280 | 0.04980 | 217s | 2.45 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 3.472 | 0.3750 | 0.5249 | 0.0017 | 0.0169 | 0.00307 | 0.05393 | 206s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 2.268 | 0.2176 | 0.2625 | 0.0008 | 0.0100 | 0.00320 | 0.05400 | 211s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.336 | 0.1128 | 0.0000 | 0.0003 | 0.0063 | 0.00291 | 0.04550 | 216s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.444 | 0.0988 | 0.0000 | 0.0002 | 0.0058 | 0.00354 | 0.05371 | 216s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.643 | 0.0829 | 0.0000 | 0.0001 | 0.0049 | 0.00480 | 0.06881 | 206s | 2.45 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.731 | 0.0719 | 0.0000 | 0.0001 | 0.0041 | 0.00616 | 0.08426 | 211s | 2.45 | completed |

### Case 06

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 1.375 | 0.3842 | 0.6834 | 0.0052 | 0.0236 | 0.00959 | 0.07453 | 237s | 3.34 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 1.006 | 0.2545 | 0.0246 | 0.0024 | 0.0136 | 0.01031 | 0.07821 | 230s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.018 | 0.1935 | 0.0000 | 0.0010 | 0.0101 | 0.01132 | 0.08234 | 232s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.008 | 0.1654 | 0.0000 | 0.0005 | 0.0087 | 0.01231 | 0.08761 | 230s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.054 | 0.1450 | 0.0000 | 0.0003 | 0.0077 | 0.01369 | 0.09459 | 238s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.132 | 0.1263 | 0.0000 | 0.0001 | 0.0068 | 0.01544 | 0.10413 | 228s | 3.34 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.186 | 0.1153 | 0.0000 | 0.0001 | 0.0062 | 0.01719 | 0.11398 | 235s | 3.34 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 6.373 | 0.8610 | 5.1062 | 0.0172 | 0.0793 | 0.01281 | 0.11220 | 196s | 2.26 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 2.679 | 0.4933 | 0.9261 | 0.0054 | 0.0349 | 0.01137 | 0.09010 | 190s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 0.999 | 0.1954 | 0.0000 | 0.0012 | 0.0103 | 0.01063 | 0.07848 | 193s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 0.991 | 0.1656 | 0.0000 | 0.0006 | 0.0088 | 0.01183 | 0.08507 | 208s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.004 | 0.1455 | 0.0000 | 0.0003 | 0.0078 | 0.01313 | 0.09213 | 201s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.096 | 0.1266 | 0.0000 | 0.0002 | 0.0068 | 0.01514 | 0.10273 | 196s | 2.26 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.141 | 0.1152 | 0.0000 | 0.0001 | 0.0062 | 0.01695 | 0.11201 | 196s | 2.26 | completed |

### Case 07

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 1.768 | 0.2879 | 0.2438 | 0.0027 | 0.0175 | 0.00499 | 0.04741 | 253s | 3.65 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 0.928 | 0.1784 | 0.0000 | 0.0011 | 0.0101 | 0.00480 | 0.04393 | 254s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.040 | 0.1516 | 0.0000 | 0.0006 | 0.0087 | 0.00611 | 0.05157 | 252s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.090 | 0.1344 | 0.0000 | 0.0003 | 0.0079 | 0.00614 | 0.05361 | 252s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.177 | 0.1229 | 0.0000 | 0.0002 | 0.0073 | 0.00684 | 0.05906 | 265s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.287 | 0.1101 | 0.0000 | 0.0001 | 0.0066 | 0.00819 | 0.06916 | 250s | 3.65 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.429 | 0.1033 | 0.0000 | 0.0001 | 0.0063 | 0.00944 | 0.07818 | 260s | 3.65 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 6.612 | 0.6598 | 2.6259 | 0.0112 | 0.0553 | 0.00844 | 0.08653 | 218s | 2.49 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 5.266 | 0.5612 | 1.8127 | 0.0054 | 0.0421 | 0.00853 | 0.08791 | 220s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 1.024 | 0.1498 | 0.0000 | 0.0006 | 0.0089 | 0.00514 | 0.04631 | 210s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.085 | 0.1339 | 0.0000 | 0.0004 | 0.0080 | 0.00581 | 0.05146 | 218s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.170 | 0.1219 | 0.0000 | 0.0002 | 0.0075 | 0.00663 | 0.05754 | 223s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.269 | 0.1099 | 0.0000 | 0.0001 | 0.0067 | 0.00796 | 0.06754 | 217s | 2.49 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.421 | 0.1030 | 0.0000 | 0.0001 | 0.0063 | 0.00929 | 0.07742 | 218s | 2.49 | completed |

### Case 08

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 6.886 | 0.4764 | 1.0963 | 0.0043 | 0.0378 | 0.00645 | 0.09147 | 281s | 4.07 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 6.006 | 0.3790 | 0.6340 | 0.0024 | 0.0288 | 0.00642 | 0.09187 | 286s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.377 | 0.1597 | 0.0000 | 0.0004 | 0.0118 | 0.00418 | 0.05202 | 280s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.127 | 0.1296 | 0.0000 | 0.0003 | 0.0103 | 0.00457 | 0.05664 | 288s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.186 | 0.1171 | 0.0000 | 0.0002 | 0.0096 | 0.00524 | 0.06292 | 287s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.335 | 0.1064 | 0.0000 | 0.0001 | 0.0090 | 0.00689 | 0.07939 | 277s | 4.07 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.397 | 0.0988 | 0.0000 | 0.0001 | 0.0085 | 0.00830 | 0.09140 | 293s | 4.07 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 11.549 | 0.6154 | 2.6500 | 0.0082 | 0.0536 | 0.00795 | 0.11990 | 240s | 2.79 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 10.223 | 0.5263 | 1.4950 | 0.0037 | 0.0376 | 0.00799 | 0.12020 | 235s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 3.170 | 0.3192 | 0.4866 | 0.0010 | 0.0198 | 0.00595 | 0.08292 | 236s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.105 | 0.1287 | 0.0000 | 0.0003 | 0.0103 | 0.00424 | 0.05271 | 236s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.185 | 0.1171 | 0.0000 | 0.0002 | 0.0097 | 0.00501 | 0.06066 | 243s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.301 | 0.1059 | 0.0000 | 0.0001 | 0.0090 | 0.00643 | 0.07455 | 240s | 2.79 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.373 | 0.0988 | 0.0000 | 0.0001 | 0.0085 | 0.00795 | 0.08893 | 252s | 2.79 | completed |

### Case 09

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 0.956 | 0.1911 | 0.0000 | 0.0018 | 0.0092 | 0.00535 | 0.03909 | 170s | 2.32 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 1.034 | 0.1640 | 0.0000 | 0.0010 | 0.0075 | 0.00555 | 0.04168 | 167s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.053 | 0.1386 | 0.0000 | 0.0005 | 0.0061 | 0.00632 | 0.04721 | 163s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.024 | 0.1244 | 0.0000 | 0.0003 | 0.0054 | 0.00712 | 0.05273 | 173s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.049 | 0.1118 | 0.0000 | 0.0002 | 0.0047 | 0.00800 | 0.05853 | 169s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.123 | 0.0981 | 0.0000 | 0.0001 | 0.0040 | 0.00948 | 0.06779 | 168s | 2.32 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.258 | 0.0898 | 0.0000 | 0.0000 | 0.0036 | 0.01086 | 0.07642 | 179s | 2.32 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 0.972 | 0.1943 | 0.0070 | 0.0023 | 0.0095 | 0.00471 | 0.03548 | 140s | 1.56 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 1.002 | 0.1646 | 0.0000 | 0.0012 | 0.0077 | 0.00522 | 0.03928 | 135s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 1.038 | 0.1387 | 0.0000 | 0.0006 | 0.0062 | 0.00605 | 0.04537 | 132s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.013 | 0.1242 | 0.0000 | 0.0003 | 0.0054 | 0.00682 | 0.05081 | 137s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.019 | 0.1120 | 0.0000 | 0.0002 | 0.0048 | 0.00776 | 0.05718 | 148s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.117 | 0.0988 | 0.0000 | 0.0001 | 0.0041 | 0.00928 | 0.06687 | 136s | 1.56 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.227 | 0.0898 | 0.0000 | 0.0000 | 0.0037 | 0.01069 | 0.07538 | 133s | 1.56 | completed |

### Case 10

| Method | alpha | TRE | SDlogJ | Folding % | BE | Diff L2 | Full 1-NCC | Mask 1-NCC | Runtime | Peak GB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INR-BSCP cps4 (FD-BE) | e4 | 0.968 | 0.1957 | 0.0092 | 0.0015 | 0.0127 | 0.00331 | 0.03508 | 219s | 3.17 | legacy alpha=1e4 source |
| INR-BSCP cps4 (FD-BE) | 2e4 | 0.991 | 0.1715 | 0.0000 | 0.0009 | 0.0112 | 0.00366 | 0.03842 | 219s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | 5e4 | 1.003 | 0.1497 | 0.0000 | 0.0005 | 0.0097 | 0.00426 | 0.04369 | 218s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | e5 | 1.057 | 0.1363 | 0.0000 | 0.0003 | 0.0088 | 0.00500 | 0.05075 | 239s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | 2e5 | 1.116 | 0.1247 | 0.0000 | 0.0002 | 0.0080 | 0.00569 | 0.05602 | 223s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | 5e5 | 1.194 | 0.1071 | 0.0000 | 0.0001 | 0.0070 | 0.00721 | 0.06975 | 230s | 3.17 | completed |
| INR-BSCP cps4 (FD-BE) | e6 | 1.265 | 0.0951 | 0.0000 | 0.0001 | 0.0063 | 0.00864 | 0.08130 | 226s | 3.17 | completed |
| D-BSCP cps4 (FD-BE) | e4 | 4.079 | 0.5038 | 1.4149 | 0.0060 | 0.0363 | 0.00569 | 0.06744 | 187s | 2.15 | legacy alpha=1e4 source |
| D-BSCP cps4 (FD-BE) | 2e4 | 1.143 | 0.2352 | 0.1615 | 0.0013 | 0.0137 | 0.00369 | 0.03987 | 183s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | 5e4 | 0.989 | 0.1494 | 0.0000 | 0.0005 | 0.0097 | 0.00403 | 0.04183 | 185s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | e5 | 1.045 | 0.1362 | 0.0000 | 0.0003 | 0.0089 | 0.00468 | 0.04777 | 197s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | 2e5 | 1.114 | 0.1242 | 0.0000 | 0.0002 | 0.0081 | 0.00551 | 0.05512 | 194s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | 5e5 | 1.197 | 0.1075 | 0.0000 | 0.0001 | 0.0070 | 0.00699 | 0.06764 | 186s | 2.15 | completed |
| D-BSCP cps4 (FD-BE) | e6 | 1.270 | 0.0950 | 0.0000 | 0.0001 | 0.0063 | 0.00850 | 0.08011 | 194s | 2.15 | completed |

## Companion CSVs

- `casewise_long.csv`: cleaned source rows for the six method/alpha combinations.
- `casewise_tre_wide.csv`: case-by-case original_snap TRE matrix.
- `casewise_stdlogj_wide.csv`: case-by-case foreground SDlogJ matrix.
- `casewise_best_by_case_method.csv`: lowest-TRE alpha per case and method.
- `casewise_delta_vs_alpha100000.csv`: per-case deltas versus alpha=e5.
