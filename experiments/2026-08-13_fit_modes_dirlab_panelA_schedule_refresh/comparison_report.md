# DIR-LAB Fit-Modes Panel A MR-D-BSCP Schedule Comparison

Acceptance: `PASS`

The replacement changes only the MR-D-BSCP stage allocation from `200/400/800/1000` (2400 epochs) to `400/400/800/900` (2500 epochs). Lower `1-NCC`, TRE, stdLogJ, and folding are better. The comparison is descriptive.

## Casewise primary endpoint

| Case | Old 1-NCC | New 1-NCC | New-old | Relative delta |
|---:|---:|---:|---:|---:|
| 01 | 0.001840 | 0.001840 | 0.000000 | 0.000% |
| 02 | 0.001341 | 0.001341 | 0.000000 | 0.004% |
| 03 | 0.001600 | 0.001600 | 0.000000 | 0.011% |
| 04 | 0.003379 | 0.003379 | 0.000001 | 0.023% |
| 05 | 0.001822 | 0.001822 | -0.000000 | -0.026% |
| 06 | 0.007831 | 0.007831 | 0.000001 | 0.011% |
| 07 | 0.003931 | 0.003924 | -0.000006 | -0.162% |
| 08 | 0.002760 | 0.002761 | 0.000001 | 0.024% |
| 09 | 0.004689 | 0.004689 | -0.000000 | -0.003% |
| 10 | 0.002965 | 0.002965 | 0.000000 | 0.000% |

## Population summary

- Old 1-NCC: `0.003216 ± 0.001950`.
- New 1-NCC: `0.003215 ± 0.001950`.
- Paired new-old mean: `-0.000000`; median: `0.000000`; range: `[-0.000006, 0.000001]`.

All statistics below use the same ten paired cases. TRE is the common registration-crop snap-to-voxel metric present in every historical and new summary; the replacement index additionally retains the new native-crop headline TRE.

| Metric | N | Old mean ± SD | Old median [Q1, Q3] | New mean ± SD | New median [Q1, Q3] | Paired new-old mean [range] |
|---|---:|---:|---:|---:|---:|---:|
| 1-NCC | 10 | 0.003216 ± 0.001950 | 0.002862 [0.001826, 0.003793] | 0.003215 ± 0.001950 | 0.002863 [0.001826, 0.003788] | -0.000000 [-0.000006, 0.000001] |
| Runtime (s) | 10 | 305.040247 ± 41.595709 | 308.918096 [281.626306, 330.798902] | 299.806592 ± 29.310741 | 299.416930 [288.353634, 322.111244] | -5.233655 [-40.912709, 51.422143] |
| Peak memory (GiB) | 10 | 5.483350 ± 1.128590 | 5.565288 [5.059545, 6.043533] | 5.392254 ± 1.109427 | 5.473138 [4.974302, 5.943998] | -0.091097 [-0.124265, -0.059515] |
| Snap TRE (mm) | 10 | 1.212317 ± 0.126661 | 1.186197 [1.132094, 1.258017] | 1.213277 ± 0.127320 | 1.187482 [1.130856, 1.261443] | 0.000960 [-0.001650, 0.004196] |
| Foreground stdLogJ | 10 | 0.185452 ± 0.064172 | 0.192313 [0.139407, 0.205210] | 0.185390 ± 0.064103 | 0.192226 [0.139418, 0.205018] | -0.000062 [-0.000259, 0.000041] |
| Foreground J≤0 (%) | 10 | 0.003705 ± 0.008542 | 0.000000 [0.000000, 0.000299] | 0.003734 ± 0.008596 | 0.000000 [0.000000, 0.000299] | 0.000029 [0.000000, 0.000152] |

## August 13 diagnostic-disabled replay cross-check

All ten August 13 records match the locked schedule, method, loss, LR, alpha, seed, spacing, and downsampling metadata. The diagnostic-enabled reruns reproduce the displacement-derived outputs exactly at the stored precision: maximum absolute differences were TRE `0`, stdLogJ `0`, folding `0`, and peak memory `0`. The full-crop diagnostic added a mean `89.9 s` runtime (median `92.6 s`; range `[56.5, 126.8] s`).

Casewise values and machine-readable population statistics are in `case_schedule_comparison.csv` and `population_schedule_comparison.csv`.

The accepted 2400-epoch cohort remains immutable historical sensitivity evidence. This report does not modify the main population index or regenerate paper diagnostics.
