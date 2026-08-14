# Fast Candidates Versus 2500-Epoch Baselines

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
