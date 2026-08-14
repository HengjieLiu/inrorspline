# OASIS Unified Acceleration Sweep: All

- Variant: `oasis_seed3_first10_unified_acceleration_sweep_alpha1000_v1`
- Output root: `${DATA_ROOT}/runs/oasis_unified_acceleration_sweep_alpha1000_v1`
- Case ranks: `0-9`
- Process model: one persistent worker per candidate, with the 10 registrations run sequentially on one assigned GPU.
- Timing convention: rank 0 is the cold case for that candidate process; ranks 1-9 are warm persistent-process cases.
- Primary accuracy readout: Dice degradation grade versus the same-method baseline.
- HD95 and ASD degradation are reported as continuous deltas, not hard cutoffs.
- Full registration time includes optimization, DVF save, evaluation, regularity metrics, and output packaging. Loaded-images-to-DVF-file time stops immediately after the dense DVF is saved.

## Dice Degradation Grades

| Grade | Meaning |
|---|---|
| `no_loss_or_gain` | Mean Dice is equal to or better than baseline. |
| `small_0_to_0p02` | Mean Dice drop is greater than 0 and no more than 0.02. |
| `moderate_0p02_to_0p05` | Mean Dice drop is greater than 0.02 and no more than 0.05. |
| `large_gt_0p05` | Mean Dice drop is greater than 0.05. |

## Candidate Summary

| Method | Candidate | Template | Dice | Dice Drop | Grade | HD95 | HD95 Delta | ASD Delta | Full Time Mean | Full Saving % | Cold Full | Warm Full | Cold DVF File | Warm DVF File | Epochs | Peak Mem GB |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D-BSCP cps4 | D4_A_fast_lr30x | A_fast_lr30x | 0.7888 | 0.0077 | small_0_to_0p02 | 2.655 | 0.343 | 0.053 | 29.92 | 68.7 | 131.30 | 18.66 | 72.21 | 11.98 | 329.1 | 1.75 |
| D-BSCP cps4 | D4_F25_lr30x | F25_lr30x | 0.7887 | 0.0078 | small_0_to_0p02 | 2.641 | 0.329 | 0.052 | 35.47 | 62.8 | 116.34 | 26.49 | 78.30 | 19.63 | 600.0 | 1.75 |
| D-BSCP cps4 | D4_A_bal_lr10x | A_bal_lr10x | 0.7946 | 0.0019 | small_0_to_0p02 | 2.365 | 0.053 | 0.009 | 43.74 | 54.2 | 143.39 | 32.67 | 78.57 | 15.30 | 476.9 | 1.75 |
| D-BSCP cps4 | D4_F40_lr10x | F40_lr10x | 0.7954 | 0.0012 | small_0_to_0p02 | 2.366 | 0.054 | 0.008 | 54.59 | 42.8 | 130.98 | 46.10 | 92.73 | 38.87 | 1000.0 | 1.75 |
| D-BSCP cps4 | D4_F60_lr3x | F60_lr3x | 0.7956 | 0.0009 | small_0_to_0p02 | 2.330 | 0.018 | 0.006 | 62.58 | 34.4 | 131.18 | 54.96 | 87.75 | 48.34 | 1500.0 | 1.75 |
| D-BSCP cps4 | D4_baseline_lr1x_full | baseline_lr1x_full | 0.7965 | 0.0000 | baseline | 2.312 | 0.000 | 0.000 | 95.47 | 0.0 | 167.81 | 87.43 | 126.34 | 80.85 | 2500.0 | 1.75 |
| SINR cps4 | S4_A_bal_lr10x | A_bal_lr10x | 0.7776 | 0.0169 | small_0_to_0p02 | 2.268 | -0.006 | 0.034 | 47.85 | 62.0 | 156.96 | 35.72 | 90.42 | 29.20 | 671.2 | 2.50 |
| SINR cps4 | S4_F40_lr10x | F40_lr10x | 0.7866 | 0.0079 | small_0_to_0p02 | 2.250 | -0.024 | 0.010 | 63.37 | 49.6 | 157.21 | 52.94 | 96.56 | 46.42 | 1000.0 | 2.50 |
| SINR cps4 | S4_F60_lr3x | F60_lr3x | 0.7943 | 0.0002 | small_0_to_0p02 | 2.178 | -0.096 | -0.008 | 83.14 | 33.9 | 181.82 | 72.17 | 123.76 | 65.44 | 1500.0 | 2.50 |
| SINR cps4 | S4_A_fast_lr30x | A_fast_lr30x | 0.5668 | 0.2277 | large_gt_0p05 | 3.891 | 1.616 | 0.849 | 31.04 | 75.3 | 142.70 | 18.64 | 83.12 | 11.90 | 250.0 | 2.50 |
| SINR cps4 | S4_F25_lr30x | F25_lr30x | 0.5640 | 0.2305 | large_gt_0p05 | 3.880 | 1.606 | 0.853 | 46.41 | 63.1 | 169.53 | 32.73 | 104.82 | 25.87 | 600.0 | 2.50 |
| SINR cps4 | S4_baseline_lr1x_full | baseline_lr1x_full | 0.7945 | 0.0000 | baseline | 2.274 | 0.000 | 0.000 | 125.82 | 0.0 | 226.28 | 114.66 | 175.00 | 107.77 | 2500.0 | 2.50 |
| D-BSCP cps2 | D2_A_fast_lr30x | A_fast_lr30x | 0.7996 | 0.0028 | small_0_to_0p02 | 2.449 | 0.171 | 0.015 | 32.58 | 66.3 | 138.42 | 20.82 | 75.36 | 8.90 | 250.0 | 2.09 |
| D-BSCP cps2 | D2_A_bal_lr10x | A_bal_lr10x | 0.8023 | 0.0001 | small_0_to_0p02 | 2.330 | 0.051 | 0.002 | 33.04 | 65.8 | 134.73 | 21.74 | 76.87 | 14.39 | 400.0 | 2.09 |
| D-BSCP cps2 | D2_F25_lr30x | F25_lr30x | 0.8001 | 0.0022 | small_0_to_0p02 | 2.435 | 0.156 | 0.011 | 39.77 | 58.9 | 145.98 | 27.97 | 87.02 | 21.20 | 600.0 | 2.09 |
| D-BSCP cps2 | D2_F40_lr10x | F40_lr10x | 0.8023 | 0.0000 | small_0_to_0p02 | 2.328 | 0.049 | 0.002 | 60.05 | 37.9 | 184.88 | 46.18 | 109.23 | 34.99 | 1000.0 | 2.09 |
| D-BSCP cps2 | D2_F60_lr3x | F60_lr3x | 0.8017 | 0.0006 | small_0_to_0p02 | 2.306 | 0.028 | 0.004 | 72.37 | 25.1 | 145.92 | 64.20 | 106.20 | 53.73 | 1500.0 | 2.09 |
| D-BSCP cps2 | D2_baseline_lr1x_full | baseline_lr1x_full | 0.8023 | 0.0000 | baseline | 2.279 | 0.000 | 0.000 | 96.65 | 0.0 | 176.28 | 87.81 | 135.45 | 80.86 | 2500.0 | 2.09 |
| MR-D-BSCP 8-4-2 | MR_A_fast_decay | A_fast_lr30x_decay | 0.8061 | 0.0008 | small_0_to_0p02 | 2.201 | -0.001 | 0.002 | 31.73 | 60.9 | 154.73 | 18.07 | 101.05 | 11.03 | 437.5 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_A_fast_lr30x | A_fast_lr30x | 0.8045 | 0.0025 | small_0_to_0p02 | 2.251 | 0.049 | 0.011 | 32.54 | 59.9 | 173.20 | 16.91 | 114.12 | 10.14 | 400.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F25_lr30x | F25_lr30x | 0.8041 | 0.0029 | small_0_to_0p02 | 2.269 | 0.067 | 0.012 | 36.12 | 55.4 | 156.07 | 22.79 | 110.78 | 15.99 | 600.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_A_bal_decay | A_bal_lr10x_decay | 0.8063 | 0.0007 | small_0_to_0p02 | 2.203 | 0.001 | 0.002 | 36.89 | 54.5 | 168.29 | 22.29 | 123.34 | 15.57 | 582.5 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_A_fast_decay_legacy | A_fast_decay_legacy | 0.8058 | 0.0012 | small_0_to_0p02 | 2.215 | 0.013 | 0.003 | 38.40 | 52.6 | 155.10 | 25.44 | 110.55 | 15.11 | 577.5 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F25_decay | F25_lr30x_decay | 0.8063 | 0.0007 | small_0_to_0p02 | 2.209 | 0.007 | 0.002 | 44.31 | 45.3 | 193.68 | 27.71 | 116.34 | 15.34 | 600.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F40_lr10x | F40_lr10x | 0.8061 | 0.0009 | small_0_to_0p02 | 2.227 | 0.025 | 0.007 | 47.39 | 41.5 | 164.52 | 34.38 | 124.81 | 27.63 | 1000.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F40_decay | F40_lr10x_decay | 0.8070 | 0.0000 | small_0_to_0p02 | 2.200 | -0.002 | 0.001 | 47.74 | 41.1 | 172.69 | 33.85 | 123.56 | 27.09 | 1000.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_A_bal_lr10x | A_bal_lr10x | 0.8061 | 0.0009 | small_0_to_0p02 | 2.219 | 0.017 | 0.004 | 48.12 | 40.6 | 201.50 | 31.08 | 130.75 | 12.76 | 502.5 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F60_lr3x | F60_lr3x | 0.8067 | 0.0003 | small_0_to_0p02 | 2.212 | 0.010 | 0.003 | 59.76 | 26.3 | 182.75 | 46.09 | 140.37 | 39.31 | 1500.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_F60_decay | F60_lr3x_decay | 0.8064 | 0.0006 | small_0_to_0p02 | 2.198 | -0.004 | 0.000 | 62.33 | 23.1 | 202.98 | 46.70 | 154.32 | 39.95 | 1500.0 | 3.86 |
| MR-D-BSCP 8-4-2 | MR_baseline_lr1x_full | baseline_lr1x_full | 0.8070 | 0.0000 | baseline | 2.202 | 0.000 | 0.000 | 81.08 | 0.0 | 188.56 | 69.14 | 151.98 | 62.45 | 2500.0 | 3.86 |
| SINR cps2 | S2_A_bal_lr10x | A_bal_lr10x | 0.7920 | 0.0072 | small_0_to_0p02 | 2.257 | 0.019 | 0.014 | 110.52 | 63.0 | 238.07 | 96.35 | 171.44 | 90.23 | 820.8 | 7.63 |
| SINR cps2 | S2_F40_lr10x | F40_lr10x | 0.7918 | 0.0074 | small_0_to_0p02 | 2.285 | 0.047 | 0.016 | 147.44 | 50.6 | 252.93 | 135.72 | 197.15 | 124.56 | 1000.0 | 7.63 |
| SINR cps2 | S2_F60_lr3x | F60_lr3x | 0.7948 | 0.0045 | small_0_to_0p02 | 2.188 | -0.049 | 0.004 | 186.00 | 37.7 | 272.16 | 176.42 | 226.03 | 165.18 | 1500.0 | 7.63 |
| SINR cps2 | S2_A_fast_lr30x | A_fast_lr30x | 0.6072 | 0.1920 | large_gt_0p05 | 3.586 | 1.349 | 0.703 | 67.40 | 77.4 | 180.44 | 54.84 | 117.21 | 48.42 | 422.0 | 7.63 |
| SINR cps2 | S2_F25_lr30x | F25_lr30x | 0.6057 | 0.1935 | large_gt_0p05 | 3.603 | 1.366 | 0.705 | 93.79 | 68.6 | 197.20 | 82.30 | 144.85 | 74.18 | 600.0 | 7.63 |
| SINR cps2 | S2_baseline_lr1x_full | baseline_lr1x_full | 0.7992 | 0.0000 | baseline | 2.237 | 0.000 | 0.000 | 298.64 | 0.0 | 428.17 | 284.25 | 370.93 | 277.89 | 2500.0 | 7.63 |

## Selection Guidance

- Start from candidates in `no_loss_or_gain` or `small_0_to_0p02` Dice grades.
- Use HD95/ASD deltas to reject candidates whose boundary behavior worsens materially, even if Dice remains acceptable.
- Prefer warm loaded-images-to-DVF-file time for deployment speed discussion, and report cold rank-0 separately.
- Peak memory is recorded from `training_peak_mem_gb`; use max over first10 as the conservative value.
