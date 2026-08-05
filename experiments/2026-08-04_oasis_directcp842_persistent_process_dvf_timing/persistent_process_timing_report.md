# OASIS Persistent-Process DVF Timing: DirectCP + SINR

- Phase: `first10`
- Case rows: 40
- Process model: one long-lived Python worker subprocess per candidate/method.
- Cold case: rank `0` inside each candidate worker. This includes CUDA/PyTorch/cuDNN first-use overhead for that worker.
- Warm cases: ranks `1-9` inside the same candidate worker. These estimate persistent-process throughput after first-use overhead has been paid.
- Amortized first10: ranks `0-9`, including the cold first case.
- DVF-only production mode excludes image/segmentation evaluation, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, and full packaging.
- SINR rows use a single fixed cps stage with no multistage knot insertion/prolongation; DirectCP `8,4,2` rows keep their existing three-stage breakdown.

Important interpretation:

- `cold_minus_warm_mean` estimates first-case overhead plus any rank0-specific case/runtime difference. It is not a pure hardware constant because rank0 is a different image pair.
- Warm ranks use the same schedule/shapes in the same process, so cuDNN algorithm choices and PyTorch/CUDA caches can be reused when applicable.
- Compare candidates primarily within the same group: cold vs cold, warm mean vs warm mean, and amortized mean vs amortized mean.

## Why Cold Rank0 Is Much Slower

This is **not** registration warm-start. Each case still starts from its own newly initialized method parameters, its own optimizer, its own image pair, and its own seed reset inside `run_one.py`. For DirectCP this means a fresh control grid; for SINR this means a fresh network/generator and spline output path. No deformation, control field, gradients, or optimizer state are reused from rank0 to later ranks.

What is warm is the long-lived Python/GPU process: CUDA context, PyTorch/CUDA allocator state, cuDNN/PyTorch kernel and algorithm-selection caches, and library/kernel first-use state for the same OASIS tensor shapes. Because ranks within a candidate use the same method, cps/schedule, and image shape, later ranks can reuse those process-level choices/caches where PyTorch supports it.

The cold-vs-warm gap is larger than expected because first-use overhead is not confined to `setup`. It is charged across several first GPU operations, including loss-volume blur/downsampling and the first optimizer-loop expansion/backward paths for each stage. The table below compares cold rank0 with the first warm rank1 in the same candidate worker.

| candidate | component | cold rank0 | warm rank1 | cold - warm | cold/warm |
|---|---|---:|---:|---:|---:|
| A_fast_decay | Full script -> DVF memory | 144.55s | 16.02s | 128.53s | 9.0x |
| A_fast_decay | Loaded images -> DVF memory | 116.93s | 15.52s | 101.41s | 7.5x |
| A_fast_decay | External wall | 144.64s | 16.10s | 128.55s | 9.0x |
| A_fast_decay | Setup | 27.62s | 496.2ms | 27.13s | 55.7x |
| A_fast_decay | Training | 116.54s | 15.52s | 101.02s | 7.5x |
| A_fast_decay | Raw optimizer loop | 104.27s | 15.50s | 88.76s | 6.7x |
| A_fast_decay | Loss-volume setup total | 11.05s | 2.6ms | 11.05s | 4258.0x |
| A_fast_decay | Total optimizer expand phase | 50.95s | 1.41s | 49.54s | 36.1x |
| A_fast_decay | Total backward/step phase | 41.36s | 7.52s | 33.84s | 5.5x |
| A_fast_decay | Stage 1 generator forward phase | 0.00s | 0.00s | 0.00s | NA |
| A_fast_decay | Stage 1 expand phase | 5.19s | 924.8ms | 4.27s | 5.6x |
| A_fast_decay | Stage 2 expand phase | 10.47s | 202.7ms | 10.27s | 51.7x |
| A_fast_decay | Stage 3 optimizer loop | 68.75s | 8.06s | 60.69s | 8.5x |
| A_fast_decay | Stage 3 expand phase | 35.29s | 282.1ms | 35.01s | 125.1x |
| A_fast_decay | Actual epochs | 600.0 | 575.0 | 25.0 | 1.0x |
| original_mrdbscp | Full script -> DVF memory | 177.41s | 62.84s | 114.57s | 2.8x |
| original_mrdbscp | Loaded images -> DVF memory | 147.71s | 62.51s | 85.20s | 2.4x |
| original_mrdbscp | External wall | 177.49s | 62.91s | 114.57s | 2.8x |
| original_mrdbscp | Setup | 29.70s | 326.8ms | 29.38s | 90.9x |
| original_mrdbscp | Training | 147.70s | 62.52s | 85.18s | 2.4x |
| original_mrdbscp | Raw optimizer loop | 135.87s | 62.48s | 73.39s | 2.2x |
| original_mrdbscp | Loss-volume setup total | 9.53s | 2.8ms | 9.53s | 3455.6x |
| original_mrdbscp | Total optimizer expand phase | 43.80s | 3.13s | 40.67s | 14.0x |
| original_mrdbscp | Total backward/step phase | 63.67s | 35.50s | 28.18s | 1.8x |
| original_mrdbscp | Stage 1 generator forward phase | 0.00s | 0.00s | 0.00s | NA |
| original_mrdbscp | Stage 1 expand phase | 3.86s | 694.8ms | 3.16s | 5.6x |
| original_mrdbscp | Stage 2 expand phase | 9.44s | 965.7ms | 8.48s | 9.8x |
| original_mrdbscp | Stage 3 optimizer loop | 93.08s | 42.03s | 51.04s | 2.2x |
| original_mrdbscp | Stage 3 expand phase | 30.49s | 1.47s | 29.02s | 20.7x |
| original_mrdbscp | Actual epochs | 2500.0 | 2500.0 | 0.0 | 1.0x |
| sinr_cps2 | Full script -> DVF memory | 356.39s | 271.29s | 85.11s | 1.3x |
| sinr_cps2 | Loaded images -> DVF memory | 327.62s | 270.86s | 56.76s | 1.2x |
| sinr_cps2 | External wall | 356.54s | 271.36s | 85.17s | 1.3x |
| sinr_cps2 | Setup | 28.78s | 425.2ms | 28.35s | 67.7x |
| sinr_cps2 | Training | 325.02s | 270.86s | 54.16s | 1.2x |
| sinr_cps2 | Raw optimizer loop | 324.21s | 270.81s | 53.40s | 1.2x |
| sinr_cps2 | Loss-volume setup total | 0.00s | 0.00s | 0.00s | NA |
| sinr_cps2 | Total generator forward phase | 70.33s | 66.94s | 3.38s | 1.1x |
| sinr_cps2 | Total optimizer expand phase | 32.76s | 2.95s | 29.81s | 11.1x |
| sinr_cps2 | Total backward/step phase | 188.98s | 172.33s | 16.64s | 1.1x |
| sinr_cps2 | Stage 1 generator forward phase | 70.33s | 66.94s | 3.38s | 1.1x |
| sinr_cps2 | Stage 1 expand phase | 32.76s | 2.95s | 29.81s | 11.1x |
| sinr_cps2 | Actual epochs | 2500.0 | 2500.0 | 0.0 | 1.0x |
| sinr_cps4 | Full script -> DVF memory | 182.49s | 107.28s | 75.21s | 1.7x |
| sinr_cps4 | Loaded images -> DVF memory | 155.60s | 106.82s | 48.77s | 1.5x |
| sinr_cps4 | External wall | 182.81s | 107.38s | 75.43s | 1.7x |
| sinr_cps4 | Setup | 26.89s | 452.1ms | 26.44s | 59.5x |
| sinr_cps4 | Training | 154.81s | 106.82s | 47.99s | 1.4x |
| sinr_cps4 | Raw optimizer loop | 153.81s | 106.80s | 47.01s | 1.4x |
| sinr_cps4 | Loss-volume setup total | 0.00s | 0.00s | 0.00s | NA |
| sinr_cps4 | Total generator forward phase | 11.99s | 10.32s | 1.68s | 1.2x |
| sinr_cps4 | Total optimizer expand phase | 27.76s | 2.92s | 24.84s | 9.5x |
| sinr_cps4 | Total backward/step phase | 84.20s | 66.02s | 18.19s | 1.3x |
| sinr_cps4 | Stage 1 generator forward phase | 11.99s | 10.32s | 1.68s | 1.2x |
| sinr_cps4 | Stage 1 expand phase | 27.76s | 2.92s | 24.84s | 9.5x |
| sinr_cps4 | Actual epochs | 2500.0 | 2500.0 | 0.0 | 1.0x |

SINR-specific interpretation:

- `sinr_cps4` and `sinr_cps2` are single-stage 2500-epoch runs at alpha `1000` and lr `1e-4`.
- SINR has no stage-to-stage knot insertion, no prolongation, and no intermediate dense diagnostic expansion in this DVF-only run.
- For SINR, the stage post-to-DVF component is final network output plus final dense B-spline expansion; the final `.npy` write is reported separately.

For `A_fast_decay`, the headline `144.55s` cold rank0 versus `16.02s` warm rank1 is therefore not explained by epochs: cold rank0 used `600` epochs and warm rank1 used `575` epochs. The large gap mainly comes from process-level first-use costs distributed through setup, loss-volume setup, expansion phases, and backward/step phases.

For practical reporting, use:

- cold single-case latency: rank0 in a fresh worker process;
- warm persistent-process throughput: mean of ranks `1-9`;
- amortized first10 latency: mean of ranks `0-9`, including the cold first case.

## Summary

| candidate | group | cases | full script -> DVF memory | loaded images -> DVF memory | full script -> DVF file | wall/case | optimizer | epochs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | cold_rank0 | 1 | 144.55s +/- 0.00s | 116.93s +/- 0.00s | 144.63s +/- 0.00s | 144.64s +/- 0.00s | 104.27s +/- 0.00s | 600.0 +/- 0.0 |
| A_fast_decay | warm_ranks1_9 | 9 | 15.07s +/- 687.2ms | 14.52s +/- 650.8ms | 15.18s +/- 690.4ms | 15.19s +/- 689.8ms | 14.50s +/- 650.2ms | 575.0 +/- 21.7 |
| A_fast_decay | amortized_ranks0_9 | 10 | 28.02s +/- 40.95s | 24.76s +/- 32.39s | 28.12s +/- 40.94s | 28.14s +/- 40.94s | 23.48s +/- 28.39s | 577.5 +/- 21.9 |
| A_fast_decay | cold_minus_warm_mean | 1 | 129.48s +/- NA | 102.41s +/- NA | 129.45s +/- NA | 129.45s +/- NA | 89.77s +/- NA | 25.0 +/- NA |
| original_mrdbscp | cold_rank0 | 1 | 177.41s +/- 0.00s | 147.71s +/- 0.00s | 177.48s +/- 0.00s | 177.49s +/- 0.00s | 135.87s +/- 0.00s | 2500.0 +/- 0.0 |
| original_mrdbscp | warm_ranks1_9 | 9 | 62.48s +/- 171.2ms | 62.07s +/- 188.3ms | 62.55s +/- 167.2ms | 62.56s +/- 167.1ms | 62.03s +/- 188.5ms | 2500.0 +/- 0.0 |
| original_mrdbscp | amortized_ranks0_9 | 10 | 73.97s +/- 36.35s | 70.63s +/- 27.08s | 74.05s +/- 36.34s | 74.06s +/- 36.34s | 69.42s +/- 23.35s | 2500.0 +/- 0.0 |
| original_mrdbscp | cold_minus_warm_mean | 1 | 114.94s +/- NA | 85.64s +/- NA | 114.92s +/- NA | 114.92s +/- NA | 73.84s +/- NA | 0.0 +/- NA |
| sinr_cps2 | cold_rank0 | 1 | 356.39s +/- 0.00s | 327.62s +/- 0.00s | 356.53s +/- 0.00s | 356.54s +/- 0.00s | 324.21s +/- 0.00s | 2500.0 +/- 0.0 |
| sinr_cps2 | warm_ranks1_9 | 9 | 271.38s +/- 146.6ms | 270.92s +/- 164.0ms | 271.45s +/- 145.5ms | 271.45s +/- 145.8ms | 270.87s +/- 164.4ms | 2500.0 +/- 0.0 |
| sinr_cps2 | amortized_ranks0_9 | 10 | 279.88s +/- 26.89s | 276.59s +/- 17.93s | 279.95s +/- 26.91s | 279.96s +/- 26.91s | 276.20s +/- 16.87s | 2500.0 +/- 0.0 |
| sinr_cps2 | cold_minus_warm_mean | 1 | 85.02s +/- NA | 56.70s +/- NA | 85.08s +/- NA | 85.08s +/- NA | 53.34s +/- NA | 0.0 +/- NA |
| sinr_cps4 | cold_rank0 | 1 | 182.49s +/- 0.00s | 155.60s +/- 0.00s | 182.81s +/- 0.00s | 182.81s +/- 0.00s | 153.81s +/- 0.00s | 2500.0 +/- 0.0 |
| sinr_cps4 | warm_ranks1_9 | 9 | 107.39s +/- 87.9ms | 106.97s +/- 93.3ms | 107.48s +/- 89.7ms | 107.49s +/- 90.4ms | 106.94s +/- 93.9ms | 2500.0 +/- 0.0 |
| sinr_cps4 | amortized_ranks0_9 | 10 | 114.90s +/- 23.75s | 111.83s +/- 15.38s | 115.02s +/- 23.82s | 115.02s +/- 23.82s | 111.63s +/- 14.82s | 2500.0 +/- 0.0 |
| sinr_cps4 | cold_minus_warm_mean | 1 | 75.10s +/- NA | 48.63s +/- NA | 75.32s +/- NA | 75.32s +/- NA | 46.86s +/- NA | 0.0 +/- NA |

## Original Minus A_fast_decay

Positive values mean `A_fast_decay` is faster/lower for that metric.

| group | metric | original | A_fast_decay | original - A_fast_decay |
|---|---|---:|---:|---:|
| cold_rank0 | full_script_to_dvf_in_memory_sec | 177.41s | 144.55s | 32.86s |
| cold_rank0 | loaded_images_to_dvf_in_memory_sec | 147.71s | 116.93s | 30.78s |
| cold_rank0 | full_script_to_dvf_file_sec | 177.48s | 144.63s | 32.85s |
| cold_rank0 | external_wall_sec | 177.49s | 144.64s | 32.85s |
| cold_rank0 | total_optimizer_loop_sec | 135.87s | 104.27s | 31.60s |
| cold_rank0 | actual_epochs | 2500.00 | 600.00 | 1900.00 |
| warm_ranks1_9 | full_script_to_dvf_in_memory_sec | 62.48s | 15.07s | 47.40s |
| warm_ranks1_9 | loaded_images_to_dvf_in_memory_sec | 62.07s | 14.52s | 47.55s |
| warm_ranks1_9 | full_script_to_dvf_file_sec | 62.55s | 15.18s | 47.37s |
| warm_ranks1_9 | external_wall_sec | 62.56s | 15.19s | 47.37s |
| warm_ranks1_9 | total_optimizer_loop_sec | 62.03s | 14.50s | 47.54s |
| warm_ranks1_9 | actual_epochs | 2500.00 | 575.00 | 1925.00 |
| amortized_ranks0_9 | full_script_to_dvf_in_memory_sec | 73.97s | 28.02s | 45.95s |
| amortized_ranks0_9 | loaded_images_to_dvf_in_memory_sec | 70.63s | 24.76s | 45.87s |
| amortized_ranks0_9 | full_script_to_dvf_file_sec | 74.05s | 28.12s | 45.92s |
| amortized_ranks0_9 | external_wall_sec | 74.06s | 28.14s | 45.92s |
| amortized_ranks0_9 | total_optimizer_loop_sec | 69.42s | 23.48s | 45.94s |
| amortized_ranks0_9 | actual_epochs | 2500.00 | 577.50 | 1922.50 |
| cold_minus_warm_mean | full_script_to_dvf_in_memory_sec | 114.94s | 129.48s | -14.54s |
| cold_minus_warm_mean | loaded_images_to_dvf_in_memory_sec | 85.64s | 102.41s | -16.77s |
| cold_minus_warm_mean | full_script_to_dvf_file_sec | 114.92s | 129.45s | -14.53s |
| cold_minus_warm_mean | external_wall_sec | 114.92s | 129.45s | -14.53s |
| cold_minus_warm_mean | total_optimizer_loop_sec | 73.84s | 89.77s | -15.93s |
| cold_minus_warm_mean | actual_epochs | 0.00 | 25.00 | -25.00 |

## SINR Minus DirectCP Runtime

Positive values mean the DirectCP candidate is faster/lower than the SINR candidate for that metric.

| group | SINR | DirectCP | metric | SINR | DirectCP | SINR - DirectCP |
|---|---|---|---|---:|---:|---:|
| cold_rank0 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 182.49s | 177.41s | 5.07s |
| cold_rank0 | sinr_cps4 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 155.60s | 147.71s | 7.89s |
| cold_rank0 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_file_sec | 182.81s | 177.48s | 5.33s |
| cold_rank0 | sinr_cps4 | original_mrdbscp | external_wall_sec | 182.81s | 177.49s | 5.33s |
| cold_rank0 | sinr_cps4 | original_mrdbscp | total_optimizer_loop_sec | 153.81s | 135.87s | 17.94s |
| cold_rank0 | sinr_cps4 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| cold_rank0 | sinr_cps4 | A_fast_decay | full_script_to_dvf_in_memory_sec | 182.49s | 144.55s | 37.94s |
| cold_rank0 | sinr_cps4 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 155.60s | 116.93s | 38.67s |
| cold_rank0 | sinr_cps4 | A_fast_decay | full_script_to_dvf_file_sec | 182.81s | 144.63s | 38.17s |
| cold_rank0 | sinr_cps4 | A_fast_decay | external_wall_sec | 182.81s | 144.64s | 38.17s |
| cold_rank0 | sinr_cps4 | A_fast_decay | total_optimizer_loop_sec | 153.81s | 104.27s | 49.54s |
| cold_rank0 | sinr_cps4 | A_fast_decay | actual_epochs | 2500.00 | 600.00 | 1900.00 |
| cold_rank0 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 356.39s | 177.41s | 178.98s |
| cold_rank0 | sinr_cps2 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 327.62s | 147.71s | 179.91s |
| cold_rank0 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_file_sec | 356.53s | 177.48s | 179.05s |
| cold_rank0 | sinr_cps2 | original_mrdbscp | external_wall_sec | 356.54s | 177.49s | 179.05s |
| cold_rank0 | sinr_cps2 | original_mrdbscp | total_optimizer_loop_sec | 324.21s | 135.87s | 188.34s |
| cold_rank0 | sinr_cps2 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| cold_rank0 | sinr_cps2 | A_fast_decay | full_script_to_dvf_in_memory_sec | 356.39s | 144.55s | 211.85s |
| cold_rank0 | sinr_cps2 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 327.62s | 116.93s | 210.69s |
| cold_rank0 | sinr_cps2 | A_fast_decay | full_script_to_dvf_file_sec | 356.53s | 144.63s | 211.90s |
| cold_rank0 | sinr_cps2 | A_fast_decay | external_wall_sec | 356.54s | 144.64s | 211.89s |
| cold_rank0 | sinr_cps2 | A_fast_decay | total_optimizer_loop_sec | 324.21s | 104.27s | 219.95s |
| cold_rank0 | sinr_cps2 | A_fast_decay | actual_epochs | 2500.00 | 600.00 | 1900.00 |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 107.39s | 62.48s | 44.91s |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 106.97s | 62.07s | 44.90s |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_file_sec | 107.48s | 62.55s | 44.93s |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | external_wall_sec | 107.49s | 62.56s | 44.93s |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | total_optimizer_loop_sec | 106.94s | 62.03s | 44.91s |
| warm_ranks1_9 | sinr_cps4 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | full_script_to_dvf_in_memory_sec | 107.39s | 15.07s | 92.31s |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 106.97s | 14.52s | 92.45s |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | full_script_to_dvf_file_sec | 107.48s | 15.18s | 92.30s |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | external_wall_sec | 107.49s | 15.19s | 92.30s |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | total_optimizer_loop_sec | 106.94s | 14.50s | 92.45s |
| warm_ranks1_9 | sinr_cps4 | A_fast_decay | actual_epochs | 2500.00 | 575.00 | 1925.00 |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 271.38s | 62.48s | 208.90s |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 270.92s | 62.07s | 208.85s |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_file_sec | 271.45s | 62.55s | 208.89s |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | external_wall_sec | 271.45s | 62.56s | 208.89s |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | total_optimizer_loop_sec | 270.87s | 62.03s | 208.84s |
| warm_ranks1_9 | sinr_cps2 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | full_script_to_dvf_in_memory_sec | 271.38s | 15.07s | 256.30s |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 270.92s | 14.52s | 256.40s |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | full_script_to_dvf_file_sec | 271.45s | 15.18s | 256.27s |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | external_wall_sec | 271.45s | 15.19s | 256.26s |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | total_optimizer_loop_sec | 270.87s | 14.50s | 256.37s |
| warm_ranks1_9 | sinr_cps2 | A_fast_decay | actual_epochs | 2500.00 | 575.00 | 1925.00 |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 114.90s | 73.97s | 40.93s |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 111.83s | 70.63s | 41.20s |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | full_script_to_dvf_file_sec | 115.02s | 74.05s | 40.97s |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | external_wall_sec | 115.02s | 74.06s | 40.97s |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | total_optimizer_loop_sec | 111.63s | 69.42s | 42.21s |
| amortized_ranks0_9 | sinr_cps4 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | full_script_to_dvf_in_memory_sec | 114.90s | 28.02s | 86.88s |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 111.83s | 24.76s | 87.07s |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | full_script_to_dvf_file_sec | 115.02s | 28.12s | 86.89s |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | external_wall_sec | 115.02s | 28.14s | 86.89s |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | total_optimizer_loop_sec | 111.63s | 23.48s | 88.15s |
| amortized_ranks0_9 | sinr_cps4 | A_fast_decay | actual_epochs | 2500.00 | 577.50 | 1922.50 |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_in_memory_sec | 279.88s | 73.97s | 205.91s |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | loaded_images_to_dvf_in_memory_sec | 276.59s | 70.63s | 205.95s |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | full_script_to_dvf_file_sec | 279.95s | 74.05s | 205.91s |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | external_wall_sec | 279.96s | 74.06s | 205.91s |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | total_optimizer_loop_sec | 276.20s | 69.42s | 206.79s |
| amortized_ranks0_9 | sinr_cps2 | original_mrdbscp | actual_epochs | 2500.00 | 2500.00 | 0.00 |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | full_script_to_dvf_in_memory_sec | 279.88s | 28.02s | 251.86s |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | loaded_images_to_dvf_in_memory_sec | 276.59s | 24.76s | 251.83s |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | full_script_to_dvf_file_sec | 279.95s | 28.12s | 251.83s |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | external_wall_sec | 279.96s | 28.14s | 251.83s |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | total_optimizer_loop_sec | 276.20s | 23.48s | 252.73s |
| amortized_ranks0_9 | sinr_cps2 | A_fast_decay | actual_epochs | 2500.00 | 577.50 | 1922.50 |

## Per-Rank Case Rows

| candidate | rank | fixed | moving | status | full script -> DVF memory | loaded images -> DVF memory | wall | optimizer | epochs |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| A_fast_decay | 0 | 0333 | 0115 | cold_rank0 | 144.55s | 116.93s | 144.64s | 104.27s | 600 |
| A_fast_decay | 1 | 0113 | 0017 | warm_persistent | 16.02s | 15.52s | 16.10s | 15.50s | 575 |
| A_fast_decay | 2 | 0037 | 0394 | warm_persistent | 13.56s | 13.19s | 13.67s | 13.18s | 550 |
| A_fast_decay | 3 | 0117 | 0322 | warm_persistent | 14.99s | 14.59s | 15.09s | 14.57s | 600 |
| A_fast_decay | 4 | 0084 | 0108 | warm_persistent | 15.26s | 14.62s | 15.39s | 14.60s | 575 |
| A_fast_decay | 5 | 0288 | 0024 | warm_persistent | 15.44s | 14.90s | 15.57s | 14.88s | 600 |
| A_fast_decay | 6 | 0194 | 0184 | warm_persistent | 14.76s | 14.17s | 14.85s | 14.15s | 550 |
| A_fast_decay | 7 | 0014 | 0149 | warm_persistent | 15.64s | 15.04s | 15.79s | 15.02s | 600 |
| A_fast_decay | 8 | 0215 | 0326 | warm_persistent | 14.97s | 14.30s | 15.11s | 14.28s | 575 |
| A_fast_decay | 9 | 0173 | 0136 | warm_persistent | 14.99s | 14.33s | 15.14s | 14.31s | 550 |
| original_mrdbscp | 0 | 0333 | 0115 | cold_rank0 | 177.41s | 147.71s | 177.49s | 135.87s | 2500 |
| original_mrdbscp | 1 | 0113 | 0017 | warm_persistent | 62.84s | 62.51s | 62.91s | 62.48s | 2500 |
| original_mrdbscp | 2 | 0037 | 0394 | warm_persistent | 62.29s | 61.86s | 62.37s | 61.82s | 2500 |
| original_mrdbscp | 3 | 0117 | 0322 | warm_persistent | 62.48s | 62.08s | 62.56s | 62.04s | 2500 |
| original_mrdbscp | 4 | 0084 | 0108 | warm_persistent | 62.53s | 62.09s | 62.62s | 62.05s | 2500 |
| original_mrdbscp | 5 | 0288 | 0024 | warm_persistent | 62.30s | 61.93s | 62.38s | 61.89s | 2500 |
| original_mrdbscp | 6 | 0194 | 0184 | warm_persistent | 62.36s | 61.95s | 62.51s | 61.92s | 2500 |
| original_mrdbscp | 7 | 0014 | 0149 | warm_persistent | 62.58s | 62.11s | 62.66s | 62.07s | 2500 |
| original_mrdbscp | 8 | 0215 | 0326 | warm_persistent | 62.51s | 62.11s | 62.61s | 62.07s | 2500 |
| original_mrdbscp | 9 | 0173 | 0136 | warm_persistent | 62.39s | 62.00s | 62.46s | 61.96s | 2500 |
| sinr_cps2 | 0 | 0333 | 0115 | cold_rank0 | 356.39s | 327.62s | 356.54s | 324.21s | 2500 |
| sinr_cps2 | 1 | 0113 | 0017 | warm_persistent | 271.29s | 270.86s | 271.36s | 270.81s | 2500 |
| sinr_cps2 | 2 | 0037 | 0394 | warm_persistent | 271.65s | 271.18s | 271.73s | 271.13s | 2500 |
| sinr_cps2 | 3 | 0117 | 0322 | warm_persistent | 271.51s | 271.12s | 271.59s | 271.08s | 2500 |
| sinr_cps2 | 4 | 0084 | 0108 | warm_persistent | 271.37s | 270.94s | 271.45s | 270.90s | 2500 |
| sinr_cps2 | 5 | 0288 | 0024 | warm_persistent | 271.17s | 270.62s | 271.26s | 270.57s | 2500 |
| sinr_cps2 | 6 | 0194 | 0184 | warm_persistent | 271.33s | 270.90s | 271.41s | 270.86s | 2500 |
| sinr_cps2 | 7 | 0014 | 0149 | warm_persistent | 271.48s | 270.93s | 271.56s | 270.88s | 2500 |
| sinr_cps2 | 8 | 0215 | 0326 | warm_persistent | 271.34s | 270.89s | 271.41s | 270.84s | 2500 |
| sinr_cps2 | 9 | 0173 | 0136 | warm_persistent | 271.24s | 270.82s | 271.32s | 270.77s | 2500 |
| sinr_cps4 | 0 | 0333 | 0115 | cold_rank0 | 182.49s | 155.60s | 182.81s | 153.81s | 2500 |
| sinr_cps4 | 1 | 0113 | 0017 | warm_persistent | 107.28s | 106.82s | 107.38s | 106.80s | 2500 |
| sinr_cps4 | 2 | 0037 | 0394 | warm_persistent | 107.49s | 107.07s | 107.61s | 107.05s | 2500 |
| sinr_cps4 | 3 | 0117 | 0322 | warm_persistent | 107.22s | 106.82s | 107.33s | 106.79s | 2500 |
| sinr_cps4 | 4 | 0084 | 0108 | warm_persistent | 107.42s | 106.99s | 107.52s | 106.96s | 2500 |
| sinr_cps4 | 5 | 0288 | 0024 | warm_persistent | 107.37s | 106.96s | 107.47s | 106.93s | 2500 |
| sinr_cps4 | 6 | 0194 | 0184 | warm_persistent | 107.44s | 107.03s | 107.53s | 107.00s | 2500 |
| sinr_cps4 | 7 | 0014 | 0149 | warm_persistent | 107.40s | 106.95s | 107.50s | 106.92s | 2500 |
| sinr_cps4 | 8 | 0215 | 0326 | warm_persistent | 107.47s | 107.06s | 107.59s | 107.03s | 2500 |
| sinr_cps4 | 9 | 0173 | 0136 | warm_persistent | 107.38s | 107.03s | 107.48s | 107.00s | 2500 |

## Artifacts

- `persistent_process_timing_report.md`
- `persistent_process_case_timing.csv`
- `persistent_process_summary.csv`
- `persistent_process_savings.csv`
- `persistent_process_pairwise_runtime_comparison.csv`
- `persistent_process_cold_warm_breakdown.csv`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
