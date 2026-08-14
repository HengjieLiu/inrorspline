# OASIS100 Selected Fast-Schedule Evaluation

- Generated: `2026-08-13 20:03:55 +0000`
- Variant: `oasis_selected_fast_first100_alpha1000_v1`
- New output root: `${DATA_ROOT}/runs/oasis_selected_fast_first100_alpha1000_v1`
- Reused first10 root: `${DATA_ROOT}/runs/oasis_unified_acceleration_sweep_alpha1000_v1`
- Combined rows: `500/500`
- Missing rows: `0`
- Runtime in this report is reference-only because GPUs were shared/occupied during this run.
- Final paper benchmark timing should be rerun under isolated persistent-process conditions.

## Accuracy And Regularity Summary

| Candidate | Method | Cases | Dice | HD95 | ASD | FG stdLogJ | FG J<=0% | Epochs | Peak Mem Max GB | Worst Mem Rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MR_A_fast_decay | MR-D-BSCP 8-4-2 | 100 | 0.8090 | 2.182 | 0.431 | 0.8770 | 3.9035 | 439.8 | 3.86 | 0 |
| D2_A_bal_lr10x | D-BSCP cps2 | 100 | 0.8048 | 2.334 | 0.459 | 0.9276 | 5.7260 | 431.1 | 2.09 | 0 |
| D4_A_bal_lr10x | D-BSCP cps4 | 100 | 0.7971 | 2.372 | 0.477 | 0.9437 | 6.0962 | 477.1 | 1.75 | 0 |
| S2_A_bal_lr10x | SINR cps2 | 100 | 0.7926 | 2.238 | 0.472 | 0.8692 | 4.4929 | 798.8 | 7.63 | 0 |
| S4_A_bal_lr10x | SINR cps4 | 100 | 0.7856 | 2.249 | 0.486 | 0.8568 | 4.2717 | 644.3 | 2.50 | 0 |

## Reference Runtime

These values are not the final benchmark. They are retained to check practical scale while other jobs may have shared the GPUs.

| Candidate | Group | Cases | Full Reg Mean | Loaded Images To DVF File Mean | External Wall Mean |
|---|---|---:|---:|---:|---:|
| MR_A_fast_decay | all_new_ranks10_99 | 90 | 25.28 | 14.83 | 25.29 |
| MR_A_fast_decay | worker_cold_cases | 4 | 142.21 | 101.60 | 142.21 |
| MR_A_fast_decay | worker_warm_cases | 86 | 19.85 | 10.79 | 19.85 |
| D2_A_bal_lr10x | all_new_ranks10_99 | 90 | 22.89 | 15.91 | 22.90 |
| D2_A_bal_lr10x | worker_cold_cases | 1 | 126.05 | 84.70 | 126.05 |
| D2_A_bal_lr10x | worker_warm_cases | 89 | 21.73 | 15.14 | 21.74 |
| D4_A_bal_lr10x | all_new_ranks10_99 | 90 | 25.24 | 16.55 | 25.24 |
| D4_A_bal_lr10x | worker_cold_cases | 1 | 118.18 | 71.99 | 118.18 |
| D4_A_bal_lr10x | worker_warm_cases | 89 | 24.19 | 15.92 | 24.20 |
| S2_A_bal_lr10x | all_new_ranks10_99 | 90 | 99.21 | 90.98 | 99.22 |
| S2_A_bal_lr10x | worker_cold_cases | 1 | 220.36 | 183.63 | 220.37 |
| S2_A_bal_lr10x | worker_warm_cases | 89 | 97.85 | 89.94 | 97.86 |
| S4_A_bal_lr10x | all_new_ranks10_99 | 90 | 40.08 | 28.98 | 40.08 |
| S4_A_bal_lr10x | worker_cold_cases | 1 | 108.06 | 70.39 | 108.06 |
| S4_A_bal_lr10x | worker_warm_cases | 89 | 39.31 | 28.51 | 39.32 |

## Fast Versus 2500-Epoch Baselines

- Fast rows: selected OASIS100 run `oasis_selected_fast_first100_alpha1000_v1` with ranks `0-9` reused from the unified first10 sweep and ranks `10-99` newly run.
- Baseline rows: OASIS seed-3 first100 extension, alpha `1000`, same method family, original 2500-epoch schedule.
- Runtime comparison uses `total_elapsed_sec_mean` and `training_elapsed_sec_mean` from the two aggregate summaries. Treat runtime as reference-only until the isolated persistent-process benchmark is rerun.
- Deltas are `fast - baseline`, so negative Dice is accuracy loss and positive HD95/ASD is worse surface distance.

| Fast Candidate | 2500-Epoch Baseline | Cases | Epochs Fast/Base | Dice Fast/Base/Delta | HD95 Fast/Base/Delta | ASD Fast/Base/Delta | Total sec Fast/Base/Saving | Training sec Fast/Base/Saving | Peak Mem GB Fast/Base/Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MR_A_fast_decay | MR-D-BSCP 8-4-2 (2500 total epochs: 400,800,1300; lr=1e-3) | 100/100 | 439.8/2500 | 0.8090/0.8096/-0.0006 | 2.182/2.168/0.014 | 0.431/0.427/0.004 | 22.42/233.93/211.50 (90.4%) | 15.22/198.69/183.47 (92.3%) | 3.86/3.93/-0.08 |
| D2_A_bal_lr10x | D-BSCP cps2 (2500 epochs; lr=1e-3) | 100/100 | 431.1/2500 | 0.8048/0.8049/-0.0002 | 2.334/2.329/0.005 | 0.459/0.456/0.003 | 22.37/166.21/143.84 (86.5%) | 16.29/107.98/91.69 (84.9%) | 2.09/2.16/-0.08 |
| D4_A_bal_lr10x | D-BSCP cps4 (2500 epochs; lr=1e-3) | 100/100 | 477.1/2500 | 0.7971/0.7994/-0.0023 | 2.372/2.287/0.085 | 0.477/0.461/0.016 | 23.46/162.24/138.78 (85.5%) | 16.96/104.46/87.50 (83.8%) | 1.75/1.83/-0.08 |
| S2_A_bal_lr10x | SINR cps2 (2500 epochs; lr=1e-4) | 100/100 | 798.8/2500 | 0.7926/0.7991/-0.0065 | 2.238/2.309/-0.071 | 0.472/0.468/0.004 | 97.82/368.95/271.14 (73.5%) | 91.63/301.64/210.01 (69.6%) | 7.63/7.72/-0.09 |
| S4_A_bal_lr10x | SINR cps4 (2500 epochs; lr=1e-4) | 100/100 | 644.3/2500 | 0.7856/0.7961/-0.0105 | 2.249/2.375/-0.126 | 0.486/0.478/0.008 | 35.92/191.39/155.48 (81.2%) | 29.51/132.57/103.06 (77.7%) | 2.50/2.58/-0.08 |

## Short Read

- `MR_A_fast_decay`: Dice delta -0.0006, HD95 delta 0.014 mm, total-time saving 90.4%, training-time saving 92.3%.
- `D2_A_bal_lr10x`: Dice delta -0.0002, HD95 delta 0.005 mm, total-time saving 86.5%, training-time saving 84.9%.
- `D4_A_bal_lr10x`: Dice delta -0.0023, HD95 delta 0.085 mm, total-time saving 85.5%, training-time saving 83.8%.
- `S2_A_bal_lr10x`: Dice delta -0.0065, HD95 delta -0.071 mm, total-time saving 73.5%, training-time saving 69.6%.
- `S4_A_bal_lr10x`: Dice delta -0.0105, HD95 delta -0.126 mm, total-time saving 81.2%, training-time saving 77.7%.

## Reuse Policy

- Ranks `0-9` are reused from the completed unified first10 acceleration sweep.
- Ranks `10-99` are newly evaluated in this selected-first100 output root.
- The combined report therefore contains 100 rows per selected candidate.
