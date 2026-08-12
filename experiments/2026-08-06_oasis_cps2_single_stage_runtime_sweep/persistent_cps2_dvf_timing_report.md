# OASIS cps2 Persistent-Process DVF Timing

- Phase: `first10`
- Candidate rows: 90
- Candidates: D2_baseline_lr1e-3_e2500, S2_baseline_lr1e-4_e2500, D2_F600_lr1e-2, D2_F400_lr3e-2, D2_A_lr1e-2, D2_A_lr3e-2, S2_F1500_lr3e-4, S2_F1000_lr1e-3, S2_A_lr1e-3
- Scope: DVF-only timing after the full-pipeline accuracy/runtime screening sweep.
- Process model: one long-lived Python worker subprocess per candidate; ranks are sequential inside that worker on GPU `0` unless changed at launch.
- Cold case: rank `0` in a fresh worker process for that candidate.
- Warm cases: ranks `1-9` in the same worker process. Warm means process/GPU warm, not registration warm-start.
- Each case still gets a fresh image pair, model/control state, optimizer state, and seed reset inside `run_one.py`.
- DVF-only excludes image/segmentation evaluation, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, and full packaging.
- D-BSCP cps2 and SINR cps2 are single-stage runs here; there is no multistage knot insertion/prolongation in these timing rows.

## Validation

- Evaluation/full-output events counted in DVF-only rows: 0
- Non-target GPU compute-app observations in pre/post snapshots: 0
- Reconciliation rows: 1800
- Reconciliation residual failures: 0
- Parent intervals with large self-unaccounted time: 0

## Timing Definitions

- `full_script_to_dvf_in_memory`: `run_one.py` script timer start to final dense DVF tensor in memory.
- `loaded_images_to_dvf_in_memory`: after OASIS case/image load to final dense DVF tensor in memory.
- `full_script_to_dvf_file`: script timer start to dense DVF `.npy` written.
- `loaded_images_to_dvf_file`: after OASIS case/image load to dense DVF `.npy` written.
- `external_wall`: persistent worker measured wall time for one in-process `run_one.main()` call.
- `optimizer_loop`: epoch work only; it is a strict subset of `training`.
- `non_loop_training`: training minus optimizer-loop time, including loss-volume/model/grid setup and final dense expansion.
- Screening Dice/HD95/runtime values come from full evaluation mode in `candidate_summary.csv`; persistent timing values are DVF-only and should not be mixed as if they include evaluation.

## Screening Gate Context

| candidate | method | promising | Dice | HD95 | Dice delta | HD95 delta | full-pipeline saving | beats cps4 ctx |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D2_baseline_lr1e-3_e2500 | D-BSCP cps2 | False | 0.8023 | 2.279 | 0.0000 | 0.000 | 0.0% | True |
| S2_baseline_lr1e-4_e2500 | SINR cps2 | False | 0.7992 | 2.237 | 0.0000 | 0.000 | 0.0% | True |
| D2_F600_lr1e-2 | D-BSCP cps2 | True | 0.8022 | 2.335 | -0.0002 | 0.056 | 25.4% | False |
| D2_F400_lr3e-2 | D-BSCP cps2 | True | 0.7998 | 2.444 | -0.0025 | 0.165 | 28.7% | False |
| D2_A_lr1e-2 | D-BSCP cps2 | True | 0.8022 | 2.328 | -0.0001 | 0.049 | 31.1% | False |
| D2_A_lr3e-2 | D-BSCP cps2 | True | 0.8004 | 2.437 | -0.0020 | 0.158 | 36.9% | False |
| S2_F1500_lr3e-4 | SINR cps2 | True | 0.7948 | 2.188 | -0.0045 | -0.049 | 29.2% | True |
| S2_F1000_lr1e-3 | SINR cps2 | True | 0.7918 | 2.285 | -0.0074 | 0.047 | 43.5% | False |
| S2_A_lr1e-3 | SINR cps2 | True | 0.7906 | 2.245 | -0.0087 | 0.007 | 40.0% | False |

## Persistent Timing Summary

| candidate | group | cases | loaded images -> DVF file | full script -> DVF file | wall/case | optimizer | non-loop training | epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D2_baseline_lr1e-3_e2500 | cold_rank0 | 1 | 156.86s +/- 0.00s | 173.02s +/- 0.00s | 173.04s +/- 0.00s | 154.64s +/- 0.00s | 1.60s +/- 0.00s | 2500.0 +/- 0.0 |
| D2_baseline_lr1e-3_e2500 | warm_ranks1_9 | 9 | 85.24s +/- 585.1ms | 85.90s +/- 598.1ms | 86.00s +/- 564.0ms | 85.01s +/- 571.8ms | 41.4ms +/- 8.4ms | 2500.0 +/- 0.0 |
| D2_baseline_lr1e-3_e2500 | amortized_ranks0_9 | 10 | 92.40s +/- 22.65s | 94.61s +/- 27.56s | 94.70s +/- 27.53s | 91.97s +/- 22.03s | 196.9ms +/- 491.9ms | 2500.0 +/- 0.0 |
| S2_baseline_lr1e-4_e2500 | cold_rank0 | 1 | 350.34s +/- 0.00s | 366.40s +/- 0.00s | 366.41s +/- 0.00s | 348.14s +/- 0.00s | 1.53s +/- 0.00s | 2500.0 +/- 0.0 |
| S2_baseline_lr1e-4_e2500 | warm_ranks1_9 | 9 | 273.24s +/- 365.1ms | 273.90s +/- 374.5ms | 273.91s +/- 374.6ms | 273.05s +/- 376.5ms | 67.0ms +/- 7.8ms | 2500.0 +/- 0.0 |
| S2_baseline_lr1e-4_e2500 | amortized_ranks0_9 | 10 | 280.95s +/- 24.38s | 283.15s +/- 29.25s | 283.16s +/- 29.25s | 280.56s +/- 23.75s | 213.1ms +/- 462.1ms | 2500.0 +/- 0.0 |
| D2_F600_lr1e-2 | cold_rank0 | 1 | 87.34s +/- 0.00s | 103.71s +/- 0.00s | 103.72s +/- 0.00s | 85.66s +/- 0.00s | 1.52s +/- 0.00s | 600.0 +/- 0.0 |
| D2_F600_lr1e-2 | warm_ranks1_9 | 9 | 20.68s +/- 268.3ms | 21.32s +/- 252.1ms | 21.33s +/- 252.6ms | 20.54s +/- 256.0ms | 14.0ms +/- 3.3ms | 600.0 +/- 0.0 |
| D2_F600_lr1e-2 | amortized_ranks0_9 | 10 | 27.34s +/- 21.08s | 29.56s +/- 26.05s | 29.57s +/- 26.05s | 27.05s +/- 20.59s | 164.1ms +/- 474.7ms | 600.0 +/- 0.0 |
| D2_F400_lr3e-2 | cold_rank0 | 1 | 83.67s +/- 0.00s | 100.68s +/- 0.00s | 100.69s +/- 0.00s | 81.50s +/- 0.00s | 1.76s +/- 0.00s | 400.0 +/- 0.0 |
| D2_F400_lr3e-2 | warm_ranks1_9 | 9 | 14.21s +/- 417.8ms | 15.40s +/- 1.37s | 15.41s +/- 1.37s | 14.05s +/- 409.6ms | 18.3ms +/- 7.5ms | 400.0 +/- 0.0 |
| D2_F400_lr3e-2 | amortized_ranks0_9 | 10 | 21.16s +/- 21.97s | 23.93s +/- 27.00s | 23.94s +/- 27.00s | 20.80s +/- 21.33s | 192.8ms +/- 551.9ms | 400.0 +/- 0.0 |
| D2_A_lr1e-2 | cold_rank0 | 1 | 80.39s +/- 0.00s | 109.98s +/- 0.00s | 109.99s +/- 0.00s | 78.12s +/- 0.00s | 1.66s +/- 0.00s | 331.0 +/- 0.0 |
| D2_A_lr1e-2 | warm_ranks1_9 | 9 | 13.71s +/- 2.01s | 14.66s +/- 1.97s | 14.67s +/- 1.97s | 13.55s +/- 2.00s | 13.7ms +/- 4.3ms | 385.1 +/- 58.3 |
| D2_A_lr1e-2 | amortized_ranks0_9 | 10 | 20.38s +/- 21.17s | 24.19s +/- 30.20s | 24.20s +/- 30.20s | 20.01s +/- 20.51s | 178.3ms +/- 520.5ms | 379.7 +/- 57.6 |
| D2_A_lr3e-2 | cold_rank0 | 1 | 76.96s +/- 0.00s | 89.99s +/- 0.00s | 90.00s +/- 0.00s | 73.52s +/- 0.00s | 1.55s +/- 0.00s | 200.0 +/- 0.0 |
| D2_A_lr3e-2 | warm_ranks1_9 | 9 | 7.95s +/- 584.6ms | 8.80s +/- 621.3ms | 8.82s +/- 621.5ms | 7.79s +/- 588.4ms | 11.9ms +/- 1.9ms | 215.9 +/- 16.2 |
| D2_A_lr3e-2 | amortized_ranks0_9 | 10 | 14.85s +/- 21.83s | 16.92s +/- 25.68s | 16.94s +/- 25.68s | 14.36s +/- 20.79s | 165.6ms +/- 486.0ms | 214.3 +/- 16.1 |
| S2_F1500_lr3e-4 | cold_rank0 | 1 | 245.26s +/- 0.00s | 259.24s +/- 0.00s | 259.25s +/- 0.00s | 243.21s +/- 0.00s | 1.50s +/- 0.00s | 1500.0 +/- 0.0 |
| S2_F1500_lr3e-4 | warm_ranks1_9 | 9 | 164.50s +/- 1.28s | 166.98s +/- 1.92s | 166.99s +/- 1.92s | 164.32s +/- 1.26s | 54.5ms +/- 6.8ms | 1500.0 +/- 0.0 |
| S2_F1500_lr3e-4 | amortized_ranks0_9 | 10 | 172.57s +/- 25.57s | 176.21s +/- 29.23s | 176.22s +/- 29.23s | 172.21s +/- 24.97s | 199.3ms +/- 458.0ms | 1500.0 +/- 0.0 |
| S2_F1000_lr1e-3 | cold_rank0 | 1 | 161.33s +/- 0.00s | 186.49s +/- 0.00s | 186.50s +/- 0.00s | 160.35s +/- 0.00s | 808.0ms +/- 0.00s | 1000.0 +/- 0.0 |
| S2_F1000_lr1e-3 | warm_ranks1_9 | 9 | 108.75s +/- 534.6ms | 109.37s +/- 585.5ms | 109.39s +/- 586.5ms | 108.59s +/- 521.2ms | 45.7ms +/- 4.0ms | 1000.0 +/- 0.0 |
| S2_F1000_lr1e-3 | amortized_ranks0_9 | 10 | 114.01s +/- 16.63s | 117.09s +/- 24.39s | 117.10s +/- 24.39s | 113.77s +/- 16.38s | 121.9ms +/- 241.1ms | 1000.0 +/- 0.0 |
| S2_A_lr1e-3 | cold_rank0 | 1 | 185.44s +/- 0.00s | 221.06s +/- 0.00s | 221.07s +/- 0.00s | 181.82s +/- 0.00s | 1.48s +/- 0.00s | 1018.0 +/- 0.0 |
| S2_A_lr1e-3 | warm_ranks1_9 | 9 | 113.61s +/- 22.94s | 114.24s +/- 23.06s | 114.34s +/- 22.93s | 113.47s +/- 22.94s | 45.3ms +/- 2.5ms | 1044.3 +/- 210.0 |
| S2_A_lr1e-3 | amortized_ranks0_9 | 10 | 120.79s +/- 31.37s | 124.92s +/- 40.17s | 125.01s +/- 40.08s | 120.31s +/- 30.58s | 188.5ms +/- 452.8ms | 1041.7 +/- 198.1 |

## Persistent Timing Versus Same-Method Baseline

Positive saving means the candidate is faster/lower than the same-method cps2 baseline.

| group | candidate | baseline | metric | baseline | candidate | saving | saving frac |
|---|---|---|---|---:|---:|---:|---:|
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 173.02s | 173.02s | 0.00s | 0.0% |
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 156.86s | 156.86s | 0.00s | 0.0% |
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 173.04s | 173.04s | 0.00s | 0.0% |
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 154.64s | 154.64s | 0.00s | 0.0% |
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.60s | 1.60s | 0.00s | 0.0% |
| cold_rank0 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 366.40s | 366.40s | 0.00s | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 350.34s | 350.34s | 0.00s | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 366.41s | 366.41s | 0.00s | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 348.14s | 348.14s | 0.00s | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.53s | 1.53s | 0.00s | 0.0% |
| cold_rank0 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 173.02s | 103.71s | 69.31s | 40.1% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 156.86s | 87.34s | 69.52s | 44.3% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 173.04s | 103.72s | 69.33s | 40.1% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 154.64s | 85.66s | 68.98s | 44.6% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.60s | 1.52s | 81.6ms | 5.1% |
| cold_rank0 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 173.02s | 100.68s | 72.34s | 41.8% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 156.86s | 83.67s | 73.19s | 46.7% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 173.04s | 100.69s | 72.35s | 41.8% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 154.64s | 81.50s | 73.15s | 47.3% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.60s | 1.76s | -166.6ms | -10.4% |
| cold_rank0 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 400.000 | 2100.000 | 84.0% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 173.02s | 109.98s | 63.04s | 36.4% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 156.86s | 80.39s | 76.47s | 48.7% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 173.04s | 109.99s | 63.05s | 36.4% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 154.64s | 78.12s | 76.52s | 49.5% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.60s | 1.66s | -62.8ms | -3.9% |
| cold_rank0 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 331.000 | 2169.000 | 86.8% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 173.02s | 89.99s | 83.03s | 48.0% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 156.86s | 76.96s | 79.90s | 50.9% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 173.04s | 90.00s | 83.04s | 48.0% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 154.64s | 73.52s | 81.12s | 52.5% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.60s | 1.55s | 47.9ms | 3.0% |
| cold_rank0 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 200.000 | 2300.000 | 92.0% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 366.40s | 259.24s | 107.16s | 29.2% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 350.34s | 245.26s | 105.08s | 30.0% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 366.41s | 259.25s | 107.16s | 29.2% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 348.14s | 243.21s | 104.92s | 30.1% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.53s | 1.50s | 25.4ms | 1.7% |
| cold_rank0 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1500.000 | 1000.000 | 40.0% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 366.40s | 186.49s | 179.91s | 49.1% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 350.34s | 161.33s | 189.02s | 54.0% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 366.41s | 186.50s | 179.91s | 49.1% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 348.14s | 160.35s | 187.78s | 53.9% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.53s | 808.0ms | 719.9ms | 47.1% |
| cold_rank0 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 366.40s | 221.06s | 145.34s | 39.7% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 350.34s | 185.44s | 164.90s | 47.1% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 366.41s | 221.07s | 145.34s | 39.7% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 348.14s | 181.82s | 166.32s | 47.8% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.53s | 1.48s | 50.7ms | 3.3% |
| cold_rank0 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1018.000 | 1482.000 | 59.3% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 85.90s | 85.90s | 0.00s | 0.0% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.24s | 85.24s | 0.00s | 0.0% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 86.00s | 86.00s | 0.00s | 0.0% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.01s | 85.01s | 0.00s | 0.0% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 41.4ms | 41.4ms | 0.00s | 0.0% |
| warm_ranks1_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 273.90s | 273.90s | 0.00s | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 273.24s | 273.24s | 0.00s | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 273.91s | 273.91s | 0.00s | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 273.05s | 273.05s | 0.00s | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 67.0ms | 67.0ms | 0.00s | 0.0% |
| warm_ranks1_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 85.90s | 21.32s | 64.58s | 75.2% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.24s | 20.68s | 64.57s | 75.7% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 86.00s | 21.33s | 64.67s | 75.2% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.01s | 20.54s | 64.47s | 75.8% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 41.4ms | 14.0ms | 27.4ms | 66.2% |
| warm_ranks1_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 85.90s | 15.40s | 70.50s | 82.1% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.24s | 14.21s | 71.03s | 83.3% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 86.00s | 15.41s | 70.59s | 82.1% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.01s | 14.05s | 70.95s | 83.5% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 41.4ms | 18.3ms | 23.1ms | 55.8% |
| warm_ranks1_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 400.000 | 2100.000 | 84.0% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 85.90s | 14.66s | 71.24s | 82.9% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.24s | 13.71s | 71.54s | 83.9% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 86.00s | 14.67s | 71.33s | 82.9% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.01s | 13.55s | 71.45s | 84.1% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 41.4ms | 13.7ms | 27.6ms | 66.8% |
| warm_ranks1_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 385.111 | 2114.889 | 84.6% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 85.90s | 8.80s | 77.09s | 89.7% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.24s | 7.95s | 77.29s | 90.7% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 86.00s | 8.82s | 77.18s | 89.7% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.01s | 7.79s | 77.22s | 90.8% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 41.4ms | 11.9ms | 29.5ms | 71.2% |
| warm_ranks1_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 215.889 | 2284.111 | 91.4% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 273.90s | 166.98s | 106.92s | 39.0% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 273.24s | 164.50s | 108.74s | 39.8% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 273.91s | 166.99s | 106.92s | 39.0% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 273.05s | 164.32s | 108.73s | 39.8% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 67.0ms | 54.5ms | 12.5ms | 18.6% |
| warm_ranks1_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1500.000 | 1000.000 | 40.0% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 273.90s | 109.37s | 164.52s | 60.1% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 273.24s | 108.75s | 164.49s | 60.2% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 273.91s | 109.39s | 164.52s | 60.1% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 273.05s | 108.59s | 164.46s | 60.2% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 67.0ms | 45.7ms | 21.3ms | 31.8% |
| warm_ranks1_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 273.90s | 114.24s | 159.66s | 58.3% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 273.24s | 113.61s | 159.63s | 58.4% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 273.91s | 114.34s | 159.57s | 58.3% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 273.05s | 113.47s | 159.58s | 58.4% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 67.0ms | 45.3ms | 21.7ms | 32.4% |
| warm_ranks1_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1044.333 | 1455.667 | 58.2% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 94.61s | 94.61s | 0.00s | 0.0% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 92.40s | 92.40s | 0.00s | 0.0% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 94.70s | 94.70s | 0.00s | 0.0% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 91.97s | 91.97s | 0.00s | 0.0% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 196.9ms | 196.9ms | 0.00s | 0.0% |
| amortized_ranks0_9 | D2_baseline_lr1e-3_e2500 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 283.15s | 283.15s | 0.00s | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 280.95s | 280.95s | 0.00s | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 283.16s | 283.16s | 0.00s | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 280.56s | 280.56s | 0.00s | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 213.1ms | 213.1ms | 0.00s | 0.0% |
| amortized_ranks0_9 | S2_baseline_lr1e-4_e2500 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 94.61s | 29.56s | 65.05s | 68.8% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 92.40s | 27.34s | 65.06s | 70.4% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 94.70s | 29.57s | 65.13s | 68.8% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 91.97s | 27.05s | 64.92s | 70.6% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 196.9ms | 164.1ms | 32.8ms | 16.7% |
| amortized_ranks0_9 | D2_F600_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 94.61s | 23.93s | 70.69s | 74.7% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 92.40s | 21.16s | 71.25s | 77.1% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 94.70s | 23.94s | 70.76s | 74.7% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 91.97s | 20.80s | 71.17s | 77.4% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 196.9ms | 192.8ms | 4.1ms | 2.1% |
| amortized_ranks0_9 | D2_F400_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 400.000 | 2100.000 | 84.0% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 94.61s | 24.19s | 70.42s | 74.4% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 92.40s | 20.38s | 72.03s | 77.9% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 94.70s | 24.20s | 70.50s | 74.4% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 91.97s | 20.01s | 71.96s | 78.2% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 196.9ms | 178.3ms | 18.6ms | 9.4% |
| amortized_ranks0_9 | D2_A_lr1e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 379.700 | 2120.300 | 84.8% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 94.61s | 16.92s | 77.69s | 82.1% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 92.40s | 14.85s | 77.55s | 83.9% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | external_wall_sec | 94.70s | 16.94s | 77.76s | 82.1% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 91.97s | 14.36s | 77.61s | 84.4% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 196.9ms | 165.6ms | 31.3ms | 15.9% |
| amortized_ranks0_9 | D2_A_lr3e-2 | D2_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 214.300 | 2285.700 | 91.4% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 283.15s | 176.21s | 106.94s | 37.8% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 280.95s | 172.57s | 108.38s | 38.6% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 283.16s | 176.22s | 106.94s | 37.8% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 280.56s | 172.21s | 108.35s | 38.6% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 213.1ms | 199.3ms | 13.8ms | 6.5% |
| amortized_ranks0_9 | S2_F1500_lr3e-4 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1500.000 | 1000.000 | 40.0% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 283.15s | 117.09s | 166.06s | 58.6% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 280.95s | 114.01s | 166.95s | 59.4% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 283.16s | 117.10s | 166.06s | 58.6% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 280.56s | 113.77s | 166.79s | 59.4% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 213.1ms | 121.9ms | 91.2ms | 42.8% |
| amortized_ranks0_9 | S2_F1000_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 283.15s | 124.92s | 158.23s | 55.9% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 280.95s | 120.79s | 160.16s | 57.0% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | external_wall_sec | 283.16s | 125.01s | 158.15s | 55.9% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 280.56s | 120.31s | 160.26s | 57.1% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 213.1ms | 188.5ms | 24.6ms | 11.5% |
| amortized_ranks0_9 | S2_A_lr1e-3 | S2_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1041.700 | 1458.300 | 58.3% |

## Per-Rank Rows

| candidate | rank | fixed | moving | status | loaded images -> DVF file | wall | optimizer | epochs |
|---|---:|---|---|---|---:|---:|---:|---:|
| D2_baseline_lr1e-3_e2500 | 0 | 0333 | 0115 | cold_rank0 | 156.86s | 173.04s | 154.64s | 2500 |
| D2_baseline_lr1e-3_e2500 | 1 | 0113 | 0017 | warm_persistent | 85.86s | 86.55s | 85.70s | 2500 |
| D2_baseline_lr1e-3_e2500 | 2 | 0037 | 0394 | warm_persistent | 83.97s | 84.63s | 83.77s | 2500 |
| D2_baseline_lr1e-3_e2500 | 3 | 0117 | 0322 | warm_persistent | 85.39s | 86.23s | 85.21s | 2500 |
| D2_baseline_lr1e-3_e2500 | 4 | 0084 | 0108 | warm_persistent | 85.60s | 86.33s | 84.93s | 2500 |
| D2_baseline_lr1e-3_e2500 | 5 | 0288 | 0024 | warm_persistent | 84.62s | 85.86s | 84.47s | 2500 |
| D2_baseline_lr1e-3_e2500 | 6 | 0194 | 0184 | warm_persistent | 85.34s | 86.04s | 85.16s | 2500 |
| D2_baseline_lr1e-3_e2500 | 7 | 0014 | 0149 | warm_persistent | 85.55s | 86.12s | 85.33s | 2500 |
| D2_baseline_lr1e-3_e2500 | 8 | 0215 | 0326 | warm_persistent | 85.56s | 86.38s | 85.36s | 2500 |
| D2_baseline_lr1e-3_e2500 | 9 | 0173 | 0136 | warm_persistent | 85.31s | 85.84s | 85.14s | 2500 |
| S2_baseline_lr1e-4_e2500 | 0 | 0333 | 0115 | cold_rank0 | 350.34s | 366.41s | 348.14s | 2500 |
| S2_baseline_lr1e-4_e2500 | 1 | 0113 | 0017 | warm_persistent | 272.99s | 273.74s | 272.84s | 2500 |
| S2_baseline_lr1e-4_e2500 | 2 | 0037 | 0394 | warm_persistent | 273.15s | 273.77s | 272.95s | 2500 |
| S2_baseline_lr1e-4_e2500 | 3 | 0117 | 0322 | warm_persistent | 273.18s | 273.87s | 272.98s | 2500 |
| S2_baseline_lr1e-4_e2500 | 4 | 0084 | 0108 | warm_persistent | 273.52s | 274.17s | 273.34s | 2500 |
| S2_baseline_lr1e-4_e2500 | 5 | 0288 | 0024 | warm_persistent | 274.02s | 274.62s | 273.85s | 2500 |
| S2_baseline_lr1e-4_e2500 | 6 | 0194 | 0184 | warm_persistent | 272.70s | 273.24s | 272.47s | 2500 |
| S2_baseline_lr1e-4_e2500 | 7 | 0014 | 0149 | warm_persistent | 273.22s | 274.05s | 273.02s | 2500 |
| S2_baseline_lr1e-4_e2500 | 8 | 0215 | 0326 | warm_persistent | 273.13s | 273.76s | 272.94s | 2500 |
| S2_baseline_lr1e-4_e2500 | 9 | 0173 | 0136 | warm_persistent | 273.27s | 273.97s | 273.10s | 2500 |
| D2_F600_lr1e-2 | 0 | 0333 | 0115 | cold_rank0 | 87.34s | 103.72s | 85.66s | 600 |
| D2_F600_lr1e-2 | 1 | 0113 | 0017 | warm_persistent | 20.42s | 21.11s | 20.33s | 600 |
| D2_F600_lr1e-2 | 2 | 0037 | 0394 | warm_persistent | 20.42s | 21.08s | 20.32s | 600 |
| D2_F600_lr1e-2 | 3 | 0117 | 0322 | warm_persistent | 20.98s | 21.56s | 20.83s | 600 |
| D2_F600_lr1e-2 | 4 | 0084 | 0108 | warm_persistent | 21.00s | 21.63s | 20.85s | 600 |
| D2_F600_lr1e-2 | 5 | 0288 | 0024 | warm_persistent | 20.92s | 21.50s | 20.75s | 600 |
| D2_F600_lr1e-2 | 6 | 0194 | 0184 | warm_persistent | 20.44s | 21.14s | 20.28s | 600 |
| D2_F600_lr1e-2 | 7 | 0014 | 0149 | warm_persistent | 20.51s | 21.14s | 20.39s | 600 |
| D2_F600_lr1e-2 | 8 | 0215 | 0326 | warm_persistent | 20.94s | 21.67s | 20.81s | 600 |
| D2_F600_lr1e-2 | 9 | 0173 | 0136 | warm_persistent | 20.48s | 21.14s | 20.34s | 600 |
| D2_F400_lr3e-2 | 0 | 0333 | 0115 | cold_rank0 | 83.67s | 100.69s | 81.50s | 400 |
| D2_F400_lr3e-2 | 1 | 0113 | 0017 | warm_persistent | 13.80s | 14.53s | 13.68s | 400 |
| D2_F400_lr3e-2 | 2 | 0037 | 0394 | warm_persistent | 13.66s | 14.29s | 13.53s | 400 |
| D2_F400_lr3e-2 | 3 | 0117 | 0322 | warm_persistent | 13.88s | 14.53s | 13.69s | 400 |
| D2_F400_lr3e-2 | 4 | 0084 | 0108 | warm_persistent | 14.26s | 15.01s | 14.09s | 400 |
| D2_F400_lr3e-2 | 5 | 0288 | 0024 | warm_persistent | 14.12s | 14.94s | 13.95s | 400 |
| D2_F400_lr3e-2 | 6 | 0194 | 0184 | warm_persistent | 14.94s | 15.72s | 14.79s | 400 |
| D2_F400_lr3e-2 | 7 | 0014 | 0149 | warm_persistent | 14.59s | 15.16s | 14.43s | 400 |
| D2_F400_lr3e-2 | 8 | 0215 | 0326 | warm_persistent | 14.12s | 15.68s | 13.95s | 400 |
| D2_F400_lr3e-2 | 9 | 0173 | 0136 | warm_persistent | 14.54s | 18.82s | 14.37s | 400 |
| D2_A_lr1e-2 | 0 | 0333 | 0115 | cold_rank0 | 80.39s | 109.99s | 78.12s | 331 |
| D2_A_lr1e-2 | 1 | 0113 | 0017 | warm_persistent | 12.79s | 15.88s | 12.65s | 350 |
| D2_A_lr1e-2 | 2 | 0037 | 0394 | warm_persistent | 11.68s | 12.52s | 11.55s | 325 |
| D2_A_lr1e-2 | 3 | 0117 | 0322 | warm_persistent | 17.64s | 18.30s | 17.50s | 491 |
| D2_A_lr1e-2 | 4 | 0084 | 0108 | warm_persistent | 12.48s | 13.07s | 12.38s | 355 |
| D2_A_lr1e-2 | 5 | 0288 | 0024 | warm_persistent | 13.70s | 14.43s | 13.54s | 399 |
| D2_A_lr1e-2 | 6 | 0194 | 0184 | warm_persistent | 12.97s | 13.66s | 12.83s | 372 |
| D2_A_lr1e-2 | 7 | 0014 | 0149 | warm_persistent | 14.67s | 15.33s | 14.50s | 410 |
| D2_A_lr1e-2 | 8 | 0215 | 0326 | warm_persistent | 11.60s | 12.46s | 11.40s | 314 |
| D2_A_lr1e-2 | 9 | 0173 | 0136 | warm_persistent | 15.83s | 16.39s | 15.64s | 450 |
| D2_A_lr3e-2 | 0 | 0333 | 0115 | cold_rank0 | 76.96s | 90.00s | 73.52s | 200 |
| D2_A_lr3e-2 | 1 | 0113 | 0017 | warm_persistent | 7.14s | 7.75s | 6.99s | 202 |
| D2_A_lr3e-2 | 2 | 0037 | 0394 | warm_persistent | 7.34s | 8.19s | 7.17s | 200 |
| D2_A_lr3e-2 | 3 | 0117 | 0322 | warm_persistent | 7.87s | 9.14s | 7.69s | 207 |
| D2_A_lr3e-2 | 4 | 0084 | 0108 | warm_persistent | 8.62s | 9.47s | 8.43s | 239 |
| D2_A_lr3e-2 | 5 | 0288 | 0024 | warm_persistent | 7.56s | 8.37s | 7.38s | 206 |
| D2_A_lr3e-2 | 6 | 0194 | 0184 | warm_persistent | 8.47s | 9.34s | 8.33s | 228 |
| D2_A_lr3e-2 | 7 | 0014 | 0149 | warm_persistent | 8.72s | 9.50s | 8.59s | 242 |
| D2_A_lr3e-2 | 8 | 0215 | 0326 | warm_persistent | 7.59s | 8.59s | 7.43s | 205 |
| D2_A_lr3e-2 | 9 | 0173 | 0136 | warm_persistent | 8.27s | 9.00s | 8.10s | 214 |
| S2_F1500_lr3e-4 | 0 | 0333 | 0115 | cold_rank0 | 245.26s | 259.25s | 243.21s | 1500 |
| S2_F1500_lr3e-4 | 1 | 0113 | 0017 | warm_persistent | 165.20s | 165.99s | 165.04s | 1500 |
| S2_F1500_lr3e-4 | 2 | 0037 | 0394 | warm_persistent | 165.11s | 169.27s | 164.92s | 1500 |
| S2_F1500_lr3e-4 | 3 | 0117 | 0322 | warm_persistent | 164.79s | 169.82s | 164.61s | 1500 |
| S2_F1500_lr3e-4 | 4 | 0084 | 0108 | warm_persistent | 165.51s | 166.90s | 165.31s | 1500 |
| S2_F1500_lr3e-4 | 5 | 0288 | 0024 | warm_persistent | 165.32s | 167.61s | 165.14s | 1500 |
| S2_F1500_lr3e-4 | 6 | 0194 | 0184 | warm_persistent | 164.93s | 167.08s | 164.72s | 1500 |
| S2_F1500_lr3e-4 | 7 | 0014 | 0149 | warm_persistent | 165.06s | 166.66s | 164.87s | 1500 |
| S2_F1500_lr3e-4 | 8 | 0215 | 0326 | warm_persistent | 162.02s | 166.41s | 161.89s | 1500 |
| S2_F1500_lr3e-4 | 9 | 0173 | 0136 | warm_persistent | 162.54s | 163.18s | 162.41s | 1500 |
| S2_F1000_lr1e-3 | 0 | 0333 | 0115 | cold_rank0 | 161.33s | 186.50s | 160.35s | 1000 |
| S2_F1000_lr1e-3 | 1 | 0113 | 0017 | warm_persistent | 107.75s | 108.17s | 107.64s | 1000 |
| S2_F1000_lr1e-3 | 2 | 0037 | 0394 | warm_persistent | 109.46s | 109.93s | 109.32s | 1000 |
| S2_F1000_lr1e-3 | 3 | 0117 | 0322 | warm_persistent | 108.77s | 109.64s | 108.57s | 1000 |
| S2_F1000_lr1e-3 | 4 | 0084 | 0108 | warm_persistent | 108.99s | 109.78s | 108.85s | 1000 |
| S2_F1000_lr1e-3 | 5 | 0288 | 0024 | warm_persistent | 108.20s | 108.74s | 108.06s | 1000 |
| S2_F1000_lr1e-3 | 6 | 0194 | 0184 | warm_persistent | 109.13s | 109.79s | 108.95s | 1000 |
| S2_F1000_lr1e-3 | 7 | 0014 | 0149 | warm_persistent | 108.43s | 109.22s | 108.29s | 1000 |
| S2_F1000_lr1e-3 | 8 | 0215 | 0326 | warm_persistent | 109.17s | 109.75s | 108.98s | 1000 |
| S2_F1000_lr1e-3 | 9 | 0173 | 0136 | warm_persistent | 108.83s | 109.44s | 108.69s | 1000 |
| S2_A_lr1e-3 | 0 | 0333 | 0115 | cold_rank0 | 185.44s | 221.07s | 181.82s | 1018 |
| S2_A_lr1e-3 | 1 | 0113 | 0017 | warm_persistent | 86.31s | 87.68s | 86.16s | 795 |
| S2_A_lr1e-3 | 2 | 0037 | 0394 | warm_persistent | 156.77s | 157.52s | 156.62s | 1438 |
| S2_A_lr1e-3 | 3 | 0117 | 0322 | warm_persistent | 106.43s | 106.87s | 106.29s | 980 |
| S2_A_lr1e-3 | 4 | 0084 | 0108 | warm_persistent | 112.35s | 112.91s | 112.22s | 1033 |
| S2_A_lr1e-3 | 5 | 0288 | 0024 | warm_persistent | 122.87s | 123.52s | 122.74s | 1133 |
| S2_A_lr1e-3 | 6 | 0194 | 0184 | warm_persistent | 126.30s | 126.96s | 126.17s | 1161 |
| S2_A_lr1e-3 | 7 | 0014 | 0149 | warm_persistent | 99.32s | 99.98s | 99.16s | 910 |
| S2_A_lr1e-3 | 8 | 0215 | 0326 | warm_persistent | 83.87s | 84.32s | 83.75s | 772 |
| S2_A_lr1e-3 | 9 | 0173 | 0136 | warm_persistent | 128.26s | 129.31s | 128.12s | 1177 |

## Artifacts

- `persistent_cps2_dvf_timing_report.md`
- `persistent_cps2_case_timing.csv`
- `persistent_cps2_summary.csv`
- `persistent_cps2_runtime_vs_baseline.csv`
- `persistent_cps2_screen_and_timing.csv`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_reconciliation.csv`
