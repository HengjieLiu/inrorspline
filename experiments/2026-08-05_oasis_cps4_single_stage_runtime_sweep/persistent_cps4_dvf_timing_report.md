# OASIS cps4 Persistent-Process DVF Timing

- Phase: `first10`
- Candidate rows: 70
- Candidates: D_baseline_lr1e-3_e2500, S_baseline_lr1e-4_e2500, D_F1000_lr3e-3, D_F600_lr1e-2, D_A_lr1e-2, S_F1000_lr1e-3, S_A_lr1e-3
- Scope: DVF-only timing after the full-pipeline accuracy/runtime screening sweep.
- Process model: one long-lived Python worker subprocess per candidate; ranks are sequential inside that worker on GPU `0` unless changed at launch.
- Cold case: rank `0` in a fresh worker process for that candidate.
- Warm cases: ranks `1-9` in the same worker process. Warm means process/GPU warm, not registration warm-start.
- Each case still gets a fresh image pair, model/control state, optimizer state, and seed reset inside `run_one.py`.
- DVF-only excludes image/segmentation evaluation, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, and full packaging.
- D-BSCP cps4 and SINR cps4 are single-stage runs here; there is no multistage knot insertion/prolongation in these timing rows.

## Validation

- Evaluation/full-output events counted in DVF-only rows: 0
- Non-target GPU compute-app observations in pre/post snapshots: 0
- Reconciliation rows: 1400
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

| candidate | method | promising | Dice | HD95 | Dice delta | HD95 delta | full-pipeline saving | beats cps2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D_baseline_lr1e-3_e2500 | D-BSCP cps4 | False | 0.7965 | 2.312 | 0.0000 | 0.000 | 0.0% | False |
| S_baseline_lr1e-4_e2500 | SINR cps4 | False | 0.7945 | 2.274 | 0.0000 | 0.000 | 0.0% | False |
| D_F1000_lr3e-3 | D-BSCP cps4 | True | 0.7958 | 2.334 | -0.0007 | 0.022 | 27.0% | False |
| D_F600_lr1e-2 | D-BSCP cps4 | True | 0.7949 | 2.380 | -0.0016 | 0.068 | 28.4% | False |
| D_A_lr1e-2 | D-BSCP cps4 | True | 0.7946 | 2.365 | -0.0019 | 0.053 | 35.9% | False |
| S_F1000_lr1e-3 | SINR cps4 | True | 0.7866 | 2.250 | -0.0079 | -0.024 | 31.7% | False |
| S_A_lr1e-3 | SINR cps4 | True | 0.7813 | 2.233 | -0.0132 | -0.041 | 29.9% | False |

## Persistent Timing Summary

| candidate | group | cases | loaded images -> DVF file | full script -> DVF file | wall/case | optimizer | non-loop training | epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D_baseline_lr1e-3_e2500 | cold_rank0 | 1 | 131.08s +/- 0.00s | 159.68s +/- 0.00s | 159.69s +/- 0.00s | 129.00s +/- 0.00s | 1.07s +/- 0.00s | 2500.0 +/- 0.0 |
| D_baseline_lr1e-3_e2500 | warm_ranks1_9 | 9 | 80.80s +/- 54.4ms | 81.28s +/- 114.2ms | 81.29s +/- 115.3ms | 80.67s +/- 51.4ms | 23.7ms +/- 0.9ms | 2500.0 +/- 0.0 |
| D_baseline_lr1e-3_e2500 | amortized_ranks0_9 | 10 | 85.83s +/- 15.90s | 89.12s +/- 24.79s | 89.13s +/- 24.79s | 85.50s +/- 15.28s | 128.2ms +/- 330.4ms | 2500.0 +/- 0.0 |
| S_baseline_lr1e-4_e2500 | cold_rank0 | 1 | 158.45s +/- 0.00s | 187.85s +/- 0.00s | 187.86s +/- 0.00s | 155.75s +/- 0.00s | 1.61s +/- 0.00s | 2500.0 +/- 0.0 |
| S_baseline_lr1e-4_e2500 | warm_ranks1_9 | 9 | 106.92s +/- 97.0ms | 107.33s +/- 129.7ms | 107.34s +/- 129.9ms | 106.80s +/- 87.6ms | 26.1ms +/- 1.6ms | 2500.0 +/- 0.0 |
| S_baseline_lr1e-4_e2500 | amortized_ranks0_9 | 10 | 112.07s +/- 16.30s | 115.39s +/- 25.46s | 115.39s +/- 25.46s | 111.70s +/- 15.48s | 184.6ms +/- 501.1ms | 2500.0 +/- 0.0 |
| D_F1000_lr3e-3 | cold_rank0 | 1 | 81.94s +/- 0.00s | 109.78s +/- 0.00s | 109.79s +/- 0.00s | 79.09s +/- 0.00s | 755.1ms +/- 0.00s | 1000.0 +/- 0.0 |
| D_F1000_lr3e-3 | warm_ranks1_9 | 9 | 31.83s +/- 27.8ms | 32.31s +/- 118.9ms | 32.32s +/- 118.6ms | 31.76s +/- 27.0ms | 9.2ms +/- 0.5ms | 1000.0 +/- 0.0 |
| D_F1000_lr3e-3 | amortized_ranks0_9 | 10 | 36.84s +/- 15.85s | 40.06s +/- 24.50s | 40.06s +/- 24.50s | 36.49s +/- 14.97s | 83.8ms +/- 235.9ms | 1000.0 +/- 0.0 |
| D_F600_lr1e-2 | cold_rank0 | 1 | 62.38s +/- 0.00s | 93.05s +/- 0.00s | 93.06s +/- 0.00s | 61.08s +/- 0.00s | 800.2ms +/- 0.00s | 600.0 +/- 0.0 |
| D_F600_lr1e-2 | warm_ranks1_9 | 9 | 19.13s +/- 40.6ms | 19.55s +/- 61.8ms | 19.56s +/- 61.9ms | 19.05s +/- 38.9ms | 6.6ms +/- 0.3ms | 600.0 +/- 0.0 |
| D_F600_lr1e-2 | amortized_ranks0_9 | 10 | 23.45s +/- 13.68s | 26.90s +/- 23.24s | 26.91s +/- 23.24s | 23.25s +/- 13.29s | 86.0ms +/- 251.0ms | 600.0 +/- 0.0 |
| D_A_lr1e-2 | cold_rank0 | 1 | 58.47s +/- 0.00s | 84.59s +/- 0.00s | 84.59s +/- 0.00s | 57.32s +/- 0.00s | 792.7ms +/- 0.00s | 510.0 +/- 0.0 |
| D_A_lr1e-2 | warm_ranks1_9 | 9 | 15.12s +/- 1.73s | 15.56s +/- 1.74s | 15.56s +/- 1.74s | 15.05s +/- 1.73s | 5.7ms +/- 0.5ms | 473.2 +/- 54.3 |
| D_A_lr1e-2 | amortized_ranks0_9 | 10 | 19.45s +/- 13.81s | 22.46s +/- 21.89s | 22.47s +/- 21.89s | 19.28s +/- 13.47s | 84.4ms +/- 248.9ms | 476.9 +/- 52.5 |
| S_F1000_lr1e-3 | cold_rank0 | 1 | 98.87s +/- 0.00s | 126.17s +/- 0.00s | 126.17s +/- 0.00s | 96.81s +/- 0.00s | 1.08s +/- 0.00s | 1000.0 +/- 0.0 |
| S_F1000_lr1e-3 | warm_ranks1_9 | 9 | 42.73s +/- 57.2ms | 43.16s +/- 101.7ms | 43.17s +/- 101.6ms | 42.63s +/- 56.9ms | 15.3ms +/- 0.8ms | 1000.0 +/- 0.0 |
| S_F1000_lr1e-3 | amortized_ranks0_9 | 10 | 48.34s +/- 17.75s | 51.46s +/- 26.25s | 51.47s +/- 26.25s | 48.04s +/- 17.13s | 121.9ms +/- 337.1ms | 1000.0 +/- 0.0 |
| S_A_lr1e-3 | cold_rank0 | 1 | 75.31s +/- 0.00s | 100.65s +/- 0.00s | 100.66s +/- 0.00s | 73.82s +/- 0.00s | 1.20s +/- 0.00s | 655.0 +/- 0.0 |
| S_A_lr1e-3 | warm_ranks1_9 | 9 | 37.04s +/- 5.97s | 37.46s +/- 5.96s | 37.47s +/- 5.96s | 36.94s +/- 5.97s | 13.8ms +/- 1.3ms | 867.7 +/- 139.1 |
| S_A_lr1e-3 | amortized_ranks0_9 | 10 | 40.86s +/- 13.35s | 43.78s +/- 20.75s | 43.79s +/- 20.76s | 40.63s +/- 12.95s | 132.0ms +/- 374.0ms | 846.4 +/- 147.4 |

## Persistent Timing Versus Same-Method Baseline

Positive saving means the candidate is faster/lower than the same-method cps4 baseline.

| group | candidate | baseline | metric | baseline | candidate | saving | saving frac |
|---|---|---|---|---:|---:|---:|---:|
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 159.68s | 159.68s | 0.00s | 0.0% |
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 131.08s | 131.08s | 0.00s | 0.0% |
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | external_wall_sec | 159.69s | 159.69s | 0.00s | 0.0% |
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 129.00s | 129.00s | 0.00s | 0.0% |
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.07s | 1.07s | 0.00s | 0.0% |
| cold_rank0 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 187.85s | 187.85s | 0.00s | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 158.45s | 158.45s | 0.00s | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | external_wall_sec | 187.86s | 187.86s | 0.00s | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 155.75s | 155.75s | 0.00s | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.61s | 1.61s | 0.00s | 0.0% |
| cold_rank0 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 159.68s | 109.78s | 49.90s | 31.3% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 131.08s | 81.94s | 49.14s | 37.5% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | external_wall_sec | 159.69s | 109.79s | 49.90s | 31.3% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 129.00s | 79.09s | 49.91s | 38.7% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.07s | 755.1ms | 313.3ms | 29.3% |
| cold_rank0 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 159.68s | 93.05s | 66.63s | 41.7% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 131.08s | 62.38s | 68.70s | 52.4% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 159.69s | 93.06s | 66.63s | 41.7% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 129.00s | 61.08s | 67.92s | 52.7% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.07s | 800.2ms | 268.2ms | 25.1% |
| cold_rank0 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 159.68s | 84.59s | 75.09s | 47.0% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 131.08s | 58.47s | 72.61s | 55.4% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 159.69s | 84.59s | 75.10s | 47.0% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 129.00s | 57.32s | 71.68s | 55.6% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 1.07s | 792.7ms | 275.7ms | 25.8% |
| cold_rank0 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 510.000 | 1990.000 | 79.6% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 187.85s | 126.17s | 61.68s | 32.8% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 158.45s | 98.87s | 59.58s | 37.6% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 187.86s | 126.17s | 61.68s | 32.8% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 155.75s | 96.81s | 58.94s | 37.8% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.61s | 1.08s | 529.5ms | 32.9% |
| cold_rank0 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 187.85s | 100.65s | 87.20s | 46.4% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 158.45s | 75.31s | 83.14s | 52.5% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 187.86s | 100.66s | 87.20s | 46.4% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 155.75s | 73.82s | 81.93s | 52.6% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 1.61s | 1.20s | 414.3ms | 25.7% |
| cold_rank0 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 655.000 | 1845.000 | 73.8% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 81.28s | 81.28s | 0.00s | 0.0% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 80.80s | 80.80s | 0.00s | 0.0% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | external_wall_sec | 81.29s | 81.29s | 0.00s | 0.0% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 80.67s | 80.67s | 0.00s | 0.0% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 23.7ms | 23.7ms | 0.00s | 0.0% |
| warm_ranks1_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 107.33s | 107.33s | 0.00s | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 106.92s | 106.92s | 0.00s | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | external_wall_sec | 107.34s | 107.34s | 0.00s | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 106.80s | 106.80s | 0.00s | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 26.1ms | 26.1ms | 0.00s | 0.0% |
| warm_ranks1_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 81.28s | 32.31s | 48.97s | 60.2% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 80.80s | 31.83s | 48.97s | 60.6% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | external_wall_sec | 81.29s | 32.32s | 48.97s | 60.2% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 80.67s | 31.76s | 48.91s | 60.6% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 23.7ms | 9.2ms | 14.5ms | 61.0% |
| warm_ranks1_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 81.28s | 19.55s | 61.73s | 76.0% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 80.80s | 19.13s | 61.67s | 76.3% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 81.29s | 19.56s | 61.73s | 75.9% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 80.67s | 19.05s | 61.62s | 76.4% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 23.7ms | 6.6ms | 17.1ms | 72.2% |
| warm_ranks1_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 81.28s | 15.56s | 65.72s | 80.9% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 80.80s | 15.12s | 65.68s | 81.3% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 81.29s | 15.56s | 65.72s | 80.9% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 80.67s | 15.05s | 65.62s | 81.3% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 23.7ms | 5.7ms | 18.1ms | 76.1% |
| warm_ranks1_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 473.222 | 2026.778 | 81.1% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 107.33s | 43.16s | 64.17s | 59.8% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 106.92s | 42.73s | 64.19s | 60.0% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 107.34s | 43.17s | 64.17s | 59.8% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 106.80s | 42.63s | 64.18s | 60.1% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 26.1ms | 15.3ms | 10.8ms | 41.6% |
| warm_ranks1_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 107.33s | 37.46s | 69.87s | 65.1% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 106.92s | 37.04s | 69.88s | 65.4% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 107.34s | 37.47s | 69.87s | 65.1% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 106.80s | 36.94s | 69.86s | 65.4% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 26.1ms | 13.8ms | 12.4ms | 47.3% |
| warm_ranks1_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 867.667 | 1632.333 | 65.3% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 89.12s | 89.12s | 0.00s | 0.0% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.83s | 85.83s | 0.00s | 0.0% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | external_wall_sec | 89.13s | 89.13s | 0.00s | 0.0% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.50s | 85.50s | 0.00s | 0.0% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 128.2ms | 128.2ms | 0.00s | 0.0% |
| amortized_ranks0_9 | D_baseline_lr1e-3_e2500 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 115.39s | 115.39s | 0.00s | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 112.07s | 112.07s | 0.00s | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | external_wall_sec | 115.39s | 115.39s | 0.00s | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 111.70s | 111.70s | 0.00s | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 184.6ms | 184.6ms | 0.00s | 0.0% |
| amortized_ranks0_9 | S_baseline_lr1e-4_e2500 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 2500.000 | 0.000 | 0.0% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 89.12s | 40.06s | 49.06s | 55.1% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.83s | 36.84s | 48.99s | 57.1% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | external_wall_sec | 89.13s | 40.06s | 49.06s | 55.0% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.50s | 36.49s | 49.01s | 57.3% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 128.2ms | 83.8ms | 44.4ms | 34.6% |
| amortized_ranks0_9 | D_F1000_lr3e-3 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 89.12s | 26.90s | 62.22s | 69.8% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.83s | 23.45s | 62.38s | 72.7% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 89.13s | 26.91s | 62.22s | 69.8% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.50s | 23.25s | 62.25s | 72.8% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 128.2ms | 86.0ms | 42.2ms | 32.9% |
| amortized_ranks0_9 | D_F600_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 600.000 | 1900.000 | 76.0% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | full_script_to_dvf_file_sec | 89.12s | 22.46s | 66.66s | 74.8% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file_sec | 85.83s | 19.45s | 66.37s | 77.3% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | external_wall_sec | 89.13s | 22.47s | 66.66s | 74.8% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_optimizer_loop_sec | 85.50s | 19.28s | 66.23s | 77.5% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | total_non_loop_training_sec | 128.2ms | 84.4ms | 43.8ms | 34.2% |
| amortized_ranks0_9 | D_A_lr1e-2 | D_baseline_lr1e-3_e2500 | actual_epochs | 2500.000 | 476.900 | 2023.100 | 80.9% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 115.39s | 51.46s | 63.92s | 55.4% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 112.07s | 48.34s | 63.73s | 56.9% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 115.39s | 51.47s | 63.93s | 55.4% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 111.70s | 48.04s | 63.65s | 57.0% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 184.6ms | 121.9ms | 62.7ms | 34.0% |
| amortized_ranks0_9 | S_F1000_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 1000.000 | 1500.000 | 60.0% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | full_script_to_dvf_file_sec | 115.39s | 43.78s | 71.60s | 62.1% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file_sec | 112.07s | 40.86s | 71.21s | 63.5% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | external_wall_sec | 115.39s | 43.79s | 71.60s | 62.1% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_optimizer_loop_sec | 111.70s | 40.63s | 71.07s | 63.6% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | total_non_loop_training_sec | 184.6ms | 132.0ms | 52.5ms | 28.5% |
| amortized_ranks0_9 | S_A_lr1e-3 | S_baseline_lr1e-4_e2500 | actual_epochs | 2500.000 | 846.400 | 1653.600 | 66.1% |

## Per-Rank Rows

| candidate | rank | fixed | moving | status | loaded images -> DVF file | wall | optimizer | epochs |
|---|---:|---|---|---|---:|---:|---:|---:|
| D_baseline_lr1e-3_e2500 | 0 | 0333 | 0115 | cold_rank0 | 131.08s | 159.69s | 129.00s | 2500 |
| D_baseline_lr1e-3_e2500 | 1 | 0113 | 0017 | warm_persistent | 80.73s | 81.21s | 80.60s | 2500 |
| D_baseline_lr1e-3_e2500 | 2 | 0037 | 0394 | warm_persistent | 80.84s | 81.40s | 80.70s | 2500 |
| D_baseline_lr1e-3_e2500 | 3 | 0117 | 0322 | warm_persistent | 80.81s | 81.26s | 80.67s | 2500 |
| D_baseline_lr1e-3_e2500 | 4 | 0084 | 0108 | warm_persistent | 80.89s | 81.54s | 80.75s | 2500 |
| D_baseline_lr1e-3_e2500 | 5 | 0288 | 0024 | warm_persistent | 80.80s | 81.27s | 80.67s | 2500 |
| D_baseline_lr1e-3_e2500 | 6 | 0194 | 0184 | warm_persistent | 80.76s | 81.25s | 80.62s | 2500 |
| D_baseline_lr1e-3_e2500 | 7 | 0014 | 0149 | warm_persistent | 80.73s | 81.19s | 80.62s | 2500 |
| D_baseline_lr1e-3_e2500 | 8 | 0215 | 0326 | warm_persistent | 80.84s | 81.24s | 80.73s | 2500 |
| D_baseline_lr1e-3_e2500 | 9 | 0173 | 0136 | warm_persistent | 80.79s | 81.21s | 80.67s | 2500 |
| S_baseline_lr1e-4_e2500 | 0 | 0333 | 0115 | cold_rank0 | 158.45s | 187.86s | 155.75s | 2500 |
| S_baseline_lr1e-4_e2500 | 1 | 0113 | 0017 | warm_persistent | 106.85s | 107.33s | 106.75s | 2500 |
| S_baseline_lr1e-4_e2500 | 2 | 0037 | 0394 | warm_persistent | 107.11s | 107.61s | 106.97s | 2500 |
| S_baseline_lr1e-4_e2500 | 3 | 0117 | 0322 | warm_persistent | 106.81s | 107.17s | 106.70s | 2500 |
| S_baseline_lr1e-4_e2500 | 4 | 0084 | 0108 | warm_persistent | 106.91s | 107.25s | 106.79s | 2500 |
| S_baseline_lr1e-4_e2500 | 5 | 0288 | 0024 | warm_persistent | 106.84s | 107.40s | 106.72s | 2500 |
| S_baseline_lr1e-4_e2500 | 6 | 0194 | 0184 | warm_persistent | 106.95s | 107.37s | 106.82s | 2500 |
| S_baseline_lr1e-4_e2500 | 7 | 0014 | 0149 | warm_persistent | 106.85s | 107.24s | 106.75s | 2500 |
| S_baseline_lr1e-4_e2500 | 8 | 0215 | 0326 | warm_persistent | 107.01s | 107.44s | 106.90s | 2500 |
| S_baseline_lr1e-4_e2500 | 9 | 0173 | 0136 | warm_persistent | 106.93s | 107.28s | 106.83s | 2500 |
| D_F1000_lr3e-3 | 0 | 0333 | 0115 | cold_rank0 | 81.94s | 109.79s | 79.09s | 1000 |
| D_F1000_lr3e-3 | 1 | 0113 | 0017 | warm_persistent | 31.82s | 32.17s | 31.75s | 1000 |
| D_F1000_lr3e-3 | 2 | 0037 | 0394 | warm_persistent | 31.81s | 32.23s | 31.74s | 1000 |
| D_F1000_lr3e-3 | 3 | 0117 | 0322 | warm_persistent | 31.86s | 32.42s | 31.79s | 1000 |
| D_F1000_lr3e-3 | 4 | 0084 | 0108 | warm_persistent | 31.84s | 32.41s | 31.77s | 1000 |
| D_F1000_lr3e-3 | 5 | 0288 | 0024 | warm_persistent | 31.82s | 32.26s | 31.75s | 1000 |
| D_F1000_lr3e-3 | 6 | 0194 | 0184 | warm_persistent | 31.85s | 32.54s | 31.78s | 1000 |
| D_F1000_lr3e-3 | 7 | 0014 | 0149 | warm_persistent | 31.87s | 32.27s | 31.78s | 1000 |
| D_F1000_lr3e-3 | 8 | 0215 | 0326 | warm_persistent | 31.80s | 32.35s | 31.72s | 1000 |
| D_F1000_lr3e-3 | 9 | 0173 | 0136 | warm_persistent | 31.80s | 32.22s | 31.72s | 1000 |
| D_F600_lr1e-2 | 0 | 0333 | 0115 | cold_rank0 | 62.38s | 93.06s | 61.08s | 600 |
| D_F600_lr1e-2 | 1 | 0113 | 0017 | warm_persistent | 19.11s | 19.48s | 19.04s | 600 |
| D_F600_lr1e-2 | 2 | 0037 | 0394 | warm_persistent | 19.09s | 19.50s | 19.01s | 600 |
| D_F600_lr1e-2 | 3 | 0117 | 0322 | warm_persistent | 19.10s | 19.59s | 19.02s | 600 |
| D_F600_lr1e-2 | 4 | 0084 | 0108 | warm_persistent | 19.16s | 19.62s | 19.06s | 600 |
| D_F600_lr1e-2 | 5 | 0288 | 0024 | warm_persistent | 19.21s | 19.63s | 19.14s | 600 |
| D_F600_lr1e-2 | 6 | 0194 | 0184 | warm_persistent | 19.12s | 19.57s | 19.05s | 600 |
| D_F600_lr1e-2 | 7 | 0014 | 0149 | warm_persistent | 19.16s | 19.61s | 19.07s | 600 |
| D_F600_lr1e-2 | 8 | 0215 | 0326 | warm_persistent | 19.11s | 19.50s | 19.04s | 600 |
| D_F600_lr1e-2 | 9 | 0173 | 0136 | warm_persistent | 19.08s | 19.49s | 19.01s | 600 |
| D_A_lr1e-2 | 0 | 0333 | 0115 | cold_rank0 | 58.47s | 84.59s | 57.32s | 510 |
| D_A_lr1e-2 | 1 | 0113 | 0017 | warm_persistent | 12.22s | 12.66s | 12.15s | 381 |
| D_A_lr1e-2 | 2 | 0037 | 0394 | warm_persistent | 16.92s | 17.49s | 16.85s | 530 |
| D_A_lr1e-2 | 3 | 0117 | 0322 | warm_persistent | 17.50s | 17.93s | 17.43s | 547 |
| D_A_lr1e-2 | 4 | 0084 | 0108 | warm_persistent | 14.50s | 15.01s | 14.44s | 454 |
| D_A_lr1e-2 | 5 | 0288 | 0024 | warm_persistent | 15.45s | 15.78s | 15.39s | 485 |
| D_A_lr1e-2 | 6 | 0194 | 0184 | warm_persistent | 16.87s | 17.32s | 16.79s | 528 |
| D_A_lr1e-2 | 7 | 0014 | 0149 | warm_persistent | 13.95s | 14.43s | 13.89s | 437 |
| D_A_lr1e-2 | 8 | 0215 | 0326 | warm_persistent | 13.89s | 14.29s | 13.82s | 435 |
| D_A_lr1e-2 | 9 | 0173 | 0136 | warm_persistent | 14.75s | 15.15s | 14.68s | 462 |
| S_F1000_lr1e-3 | 0 | 0333 | 0115 | cold_rank0 | 98.87s | 126.17s | 96.81s | 1000 |
| S_F1000_lr1e-3 | 1 | 0113 | 0017 | warm_persistent | 42.64s | 43.02s | 42.53s | 1000 |
| S_F1000_lr1e-3 | 2 | 0037 | 0394 | warm_persistent | 42.68s | 43.21s | 42.58s | 1000 |
| S_F1000_lr1e-3 | 3 | 0117 | 0322 | warm_persistent | 42.81s | 43.26s | 42.70s | 1000 |
| S_F1000_lr1e-3 | 4 | 0084 | 0108 | warm_persistent | 42.74s | 43.27s | 42.63s | 1000 |
| S_F1000_lr1e-3 | 5 | 0288 | 0024 | warm_persistent | 42.74s | 43.22s | 42.64s | 1000 |
| S_F1000_lr1e-3 | 6 | 0194 | 0184 | warm_persistent | 42.69s | 43.06s | 42.58s | 1000 |
| S_F1000_lr1e-3 | 7 | 0014 | 0149 | warm_persistent | 42.81s | 43.27s | 42.70s | 1000 |
| S_F1000_lr1e-3 | 8 | 0215 | 0326 | warm_persistent | 42.71s | 43.05s | 42.60s | 1000 |
| S_F1000_lr1e-3 | 9 | 0173 | 0136 | warm_persistent | 42.74s | 43.18s | 42.64s | 1000 |
| S_A_lr1e-3 | 0 | 0333 | 0115 | cold_rank0 | 75.31s | 100.66s | 73.82s | 655 |
| S_A_lr1e-3 | 1 | 0113 | 0017 | warm_persistent | 28.32s | 28.88s | 28.23s | 665 |
| S_A_lr1e-3 | 2 | 0037 | 0394 | warm_persistent | 36.87s | 37.24s | 36.78s | 866 |
| S_A_lr1e-3 | 3 | 0117 | 0322 | warm_persistent | 35.64s | 36.05s | 35.54s | 835 |
| S_A_lr1e-3 | 4 | 0084 | 0108 | warm_persistent | 47.19s | 47.66s | 47.09s | 1104 |
| S_A_lr1e-3 | 5 | 0288 | 0024 | warm_persistent | 45.57s | 46.00s | 45.47s | 1067 |
| S_A_lr1e-3 | 6 | 0194 | 0184 | warm_persistent | 35.53s | 36.00s | 35.43s | 832 |
| S_A_lr1e-3 | 7 | 0014 | 0149 | warm_persistent | 33.61s | 33.97s | 33.51s | 787 |
| S_A_lr1e-3 | 8 | 0215 | 0326 | warm_persistent | 32.95s | 33.37s | 32.86s | 772 |
| S_A_lr1e-3 | 9 | 0173 | 0136 | warm_persistent | 37.64s | 38.07s | 37.56s | 881 |

## Artifacts

- `persistent_cps4_dvf_timing_report.md`
- `persistent_cps4_case_timing.csv`
- `persistent_cps4_summary.csv`
- `persistent_cps4_runtime_vs_baseline.csv`
- `persistent_cps4_screen_and_timing.csv`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_reconciliation.csv`
