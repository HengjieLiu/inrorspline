# OASIS Runtime-Reduction Selection Summary

## Overview

The selected OASIS100 accelerated schedules come from the unified first10 acceleration sweep in `experiments/2026-08-13_oasis_unified_acceleration_sweep/`. That sweep tested a common acceleration template across MR-D-BSCP, D-BSCP, and SINR: fixed shorter epoch budgets with higher learning rates, plus adaptive early stopping schedules with higher learning rates.

The primary accuracy readout was Dice degradation versus each method's own baseline. Dice degradation was graded rather than treated as a hard cutoff:

- `no_loss_or_gain`: Dice equal to or better than baseline.
- `small_0_to_0p02`: Dice drop greater than 0 and no more than 0.02.
- `moderate_0p02_to_0p05`: Dice drop greater than 0.02 and no more than 0.05.
- `large_gt_0p05`: Dice drop greater than 0.05.

HD95 and ASD degradation were reported continuously and used to judge whether a schedule looked robust enough for full OASIS100 validation. Runtime and peak memory were also reported. The final picks favor consistency where possible: D-BSCP cps2/cps4 both use `A_bal_lr10x`, SINR cps2/cps4 both use `A_bal_lr10x`, and MR-D-BSCP uses the best multistage-specific decayed-LR schedule.

## Final Selected Versions

| Method Family | Final Selected Candidate | Strategy |
|---|---|---|
| MR-D-BSCP `8-4-2` | `MR_A_fast_decay` | adaptive fast early stopping with decayed stage LR `3e-2,1e-2,3e-3` |
| D-BSCP cps2 | `D2_A_bal_lr10x` | adaptive balanced early stopping with `10x` method-native LR |
| D-BSCP cps4 | `D4_A_bal_lr10x` | adaptive balanced early stopping with `10x` method-native LR |
| SINR cps2 | `S2_A_bal_lr10x` | adaptive balanced early stopping with `10x` method-native LR |
| SINR cps4 | `S4_A_bal_lr10x` | adaptive balanced early stopping with `10x` method-native LR |

## MR-D-BSCP `8-4-2`

Top-three schedules considered:

| Candidate | Dice | Dice Drop | HD95 | HD95 Delta | ASD Delta | Full Runtime Saving | Warm DVF File | Mean Epochs | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MR_A_fast_decay` | 0.8061 | 0.0008 | 2.201 | -0.001 | 0.002 | 60.9% | 11.0s | 437.5 | 3.86 GB |
| `MR_A_fast_lr30x` | 0.8045 | 0.0025 | 2.251 | 0.049 | 0.011 | 59.9% | 10.1s | 400.0 | 3.86 GB |
| `MR_A_bal_decay` | 0.8063 | 0.0007 | 2.203 | 0.001 | 0.002 | 54.5% | 15.6s | 582.5 | 3.86 GB |

`MR_A_fast_decay` was selected because it combined the strongest full-runtime saving with negligible Dice/HD95 change. Compared with `MR_A_fast_lr30x`, the decayed LR schedule was slightly more stable in HD95. Compared with `MR_A_bal_decay`, it stopped earlier and saved more time while maintaining essentially the same first10 accuracy.

## D-BSCP cps2

Top-three schedules considered:

| Candidate | Dice | Dice Drop | HD95 | HD95 Delta | ASD Delta | Full Runtime Saving | Warm DVF File | Mean Epochs | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `D2_A_bal_lr10x` | 0.8023 | 0.0001 | 2.330 | 0.051 | 0.002 | 65.8% | 14.4s | 400.0 | 2.09 GB |
| `D2_A_fast_lr30x` | 0.7996 | 0.0028 | 2.449 | 0.171 | 0.015 | 66.3% | 8.9s | 250.0 | 2.09 GB |
| `D2_F40_lr10x` | 0.8023 | 0.0000 | 2.328 | 0.049 | 0.002 | 37.9% | 35.0s | 1000.0 | 2.09 GB |

`D2_A_bal_lr10x` was selected because it preserved Dice and HD95 almost as well as the conservative fixed schedule while giving nearly the same full-runtime saving as the more aggressive `A_fast_lr30x`. This also keeps D-BSCP cps2 consistent with D-BSCP cps4.

## D-BSCP cps4

Top-three schedules considered:

| Candidate | Dice | Dice Drop | HD95 | HD95 Delta | ASD Delta | Full Runtime Saving | Warm DVF File | Mean Epochs | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `D4_A_bal_lr10x` | 0.7946 | 0.0019 | 2.365 | 0.053 | 0.009 | 54.2% | 15.3s | 476.9 | 1.75 GB |
| `D4_A_fast_lr30x` | 0.7888 | 0.0077 | 2.655 | 0.343 | 0.053 | 68.7% | 12.0s | 329.1 | 1.75 GB |
| `D4_F40_lr10x` | 0.7954 | 0.0012 | 2.366 | 0.054 | 0.008 | 42.8% | 38.9s | 1000.0 | 1.75 GB |

`D4_A_bal_lr10x` was selected because it is much cleaner in HD95 than `D4_A_fast_lr30x`, while retaining substantial runtime reduction. It also matches the D-BSCP cps2 schedule, making the D-BSCP acceleration story consistent across resolutions.

## SINR cps2

Top-three schedules considered:

| Candidate | Dice | Dice Drop | HD95 | HD95 Delta | ASD Delta | Full Runtime Saving | Warm DVF File | Mean Epochs | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S2_A_bal_lr10x` | 0.7920 | 0.0072 | 2.257 | 0.019 | 0.014 | 63.0% | 90.2s | 820.8 | 7.63 GB |
| `S2_F40_lr10x` | 0.7918 | 0.0074 | 2.285 | 0.047 | 0.016 | 50.6% | 124.6s | 1000.0 | 7.63 GB |
| `S2_F60_lr3x` | 0.7948 | 0.0045 | 2.188 | -0.049 | 0.004 | 37.7% | 165.2s | 1500.0 | 7.63 GB |

`S2_A_bal_lr10x` was selected because it gave the best speed among the acceptable SINR cps2 schedules while keeping HD95 degradation small. The `30x` LR schedules were rejected because they produced large Dice drops and large HD95 increases.

## SINR cps4

Top-three schedules considered:

| Candidate | Dice | Dice Drop | HD95 | HD95 Delta | ASD Delta | Full Runtime Saving | Warm DVF File | Mean Epochs | Peak Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S4_A_bal_lr10x` | 0.7776 | 0.0169 | 2.268 | -0.006 | 0.034 | 62.0% | 29.2s | 671.2 | 2.50 GB |
| `S4_F40_lr10x` | 0.7866 | 0.0079 | 2.250 | -0.024 | 0.010 | 49.6% | 46.4s | 1000.0 | 2.50 GB |
| `S4_F60_lr3x` | 0.7943 | 0.0002 | 2.178 | -0.096 | -0.008 | 33.9% | 65.4s | 1500.0 | 2.50 GB |

`S4_A_bal_lr10x` was selected to keep SINR cps2/cps4 consistent and because it gave the strongest acceptable speedup. It has a larger Dice drop than `F40_lr10x` and `F60_lr3x`, so this first100 validation is important for checking whether the first10 tradeoff generalizes.

## Selection Rationale

The final set is intentionally simple:

- D-BSCP uses the same `A_bal_lr10x` schedule for cps2 and cps4.
- SINR uses the same `A_bal_lr10x` schedule for cps2 and cps4.
- MR-D-BSCP keeps a multistage-specific decayed LR schedule because this was the best-performing practical acceleration for the `8-4-2` path.

This gives a defensible paper narrative: all fast variants preserve the original registration objective and regularization weight, increase the optimizer learning rate relative to the method-native baseline, and use adaptive early stopping. The only method-specific addition is the MR-D-BSCP stage-decayed LR schedule, which matches its multiresolution structure.
