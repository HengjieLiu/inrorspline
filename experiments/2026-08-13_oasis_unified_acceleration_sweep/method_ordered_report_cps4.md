# OASIS Unified Acceleration Sweep: cps4 Context

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

## Selection Guidance

- Start from candidates in `no_loss_or_gain` or `small_0_to_0p02` Dice grades.
- Use HD95/ASD deltas to reject candidates whose boundary behavior worsens materially, even if Dice remains acceptable.
- Prefer warm loaded-images-to-DVF-file time for deployment speed discussion, and report cold rank-0 separately.
- Peak memory is recorded from `training_peak_mem_gb`; use max over first10 as the conservative value.
