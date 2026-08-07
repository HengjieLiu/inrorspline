# OASIS DVF-Only Runtime Timing

- Phase: `first10`
- Output root: `<external-run-root>/runs/oasis_cps4_single_stage_persistent_dvf_timing_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-05_oasis_cps4_single_stage_runtime_sweep`
- Completed command rows: 70
- Timing JSON rows: 70/70
- Dense DVF files present: 70/70
- Failed subprocesses: 0
- Non-target GPU compute-app observations: 0
- Evaluation/full-output events counted in DVF-only runs: 0

## Reconciliation

- Residual failures: 0
- Parent intervals with large self-unaccounted time: 0

## Timing Hierarchy And Definitions

Conceptual hierarchy:

```text
external_wall
  script_total
    setup
      device
      seed
      case_load
      case_validate_print
      sampler_init
    method_execution
      initial_dense_grid / small setup
      training
        stage 1
          pre_stage_setup
          optimizer_loop
          post_stage_processing
        stage 2
          pre_stage_setup
          optimizer_loop
          post_stage_processing
        stage 3
          pre_stage_setup
          optimizer_loop
          post_stage_processing
            final_dense_expand = dvf_in_memory_ready
      dvf_file_write = dvf_file_ready
      small metadata assembly
    post_dvf_metadata_write
  external_uninstrumented
```

Definitions:

- `external_wall`: launcher-measured subprocess wall time.
- `script_total`: in-script measured wall time from `run_one.py` timing start to timing finalization.
- `full_script_to_dvf_in_memory`: script timer start to the final dense displacement tensor being ready in memory.
- `loaded_images_to_dvf_in_memory`: after case/image load completes to the in-memory dense DVF endpoint.
- `full_script_to_dvf_file`: script timer start to the dense DVF `.npy` being written.
- `loaded_images_to_dvf_file`: after case/image load completes to the dense DVF `.npy` being written.
- `dvf_file_write`: the timed `numpy.save` write of the normalized-grid dense displacement array only.
- `optimizer_loop`: epoch-loop work. It is a strict subset of `training`.
- `training`: the optimization block, including non-loop stage work; DirectCP uses three stages and SINR uses one.
- `non_loop_training`: stage setup, pure knot insertion, optional diagnostic dense checks/saves, final output dense expansion, control stats, and other non-epoch work.
- `external_uninstrumented`: process startup/import/argparse/exit and timing-artifact overhead outside `script_total`.
- For SINR cps4/cps2, the hierarchy collapses to one fixed-cps stage: no prolongation, no knot insertion, and no intermediate dense diagnostic expansion.

Inclusion/exclusion rules:

- DVF-only headline times include setup, image/case load for full-script starts, method optimization, final dense expansion, and optionally dense DVF file write.
- DVF-only headline times do not include image warp evaluation, segmentation warp, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, or full CSV packaging.
- Previous `fixed/io known` is not the DVF-only headline because it included evaluation and regularity/output packaging that are intentionally skipped here.
- Intermediate dense expansion is not required to start the next stage. Production DVF timing skips it by default.
- `pure_knot_insertion` means control-grid knot insertion only. Optional preservation dense expansion and metrics are reported separately.
- SINR has no multistage knot insertion/prolongation; its post-to-DVF component is final model output plus final dense B-spline expansion.
- Dense expansion and knot insertion are already PyTorch tensor operations. In normal GPU DirectCP runs they operate on CUDA tensors, so the main avoidable cost here is diagnostic dense expansion/preservation checking, not missing GPU acceleration.
- The final-stage `dvf_in_memory_ready` mark is recorded immediately after final output dense expansion. Final control-summary stats may occur after that mark, so they are part of total `non_loop_training` but not part of the in-memory DVF endpoint.

Accounting formulas:

```text
optimizer_loop is a strict subset of training.

training =
  total_optimizer_loop
  + total_non_loop_training

full_script_to_dvf_in_memory =
  setup
  + initial_dense_grid / small setup
  + training_to_final_dense_expand

full_script_to_dvf_file =
  full_script_to_dvf_in_memory
  + dvf_file_write
  + small file-prep overhead

loaded_images_to_dvf_in_memory =
  full_script_to_dvf_in_memory
  - time_before_case_load_complete

loaded_images_to_dvf_file =
  full_script_to_dvf_file
  - time_before_case_load_complete

external_wall =
  script_total
  + external_uninstrumented
```

## Headline Times

| candidate | full script -> DVF memory | loaded images -> DVF memory | full script -> DVF file | loaded images -> DVF file | DVF file write | external wall | external uninstrumented | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_A_lr1e-2 | 84.47s | 58.36s | 84.59s | 58.47s | 29.3ms | 84.59s | 4.1ms | 510.0 |
| D_A_lr1e-2 | 12.59s | 12.16s | 12.65s | 12.22s | 31.4ms | 12.66s | 4.0ms | 381.0 |
| D_A_lr1e-2 | 17.42s | 16.86s | 17.48s | 16.92s | 30.5ms | 17.49s | 5.2ms | 530.0 |
| D_A_lr1e-2 | 17.86s | 17.44s | 17.93s | 17.50s | 31.5ms | 17.93s | 2.4ms | 547.0 |
| D_A_lr1e-2 | 14.95s | 14.44s | 15.01s | 14.50s | 29.4ms | 15.01s | 3.8ms | 454.0 |
| D_A_lr1e-2 | 15.72s | 15.40s | 15.77s | 15.45s | 29.9ms | 15.78s | 3.9ms | 485.0 |
| D_A_lr1e-2 | 17.23s | 16.80s | 17.31s | 16.87s | 50.2ms | 17.32s | 4.1ms | 528.0 |
| D_A_lr1e-2 | 14.36s | 13.89s | 14.42s | 13.95s | 32.6ms | 14.43s | 3.8ms | 437.0 |
| D_A_lr1e-2 | 14.22s | 13.83s | 14.29s | 13.89s | 28.9ms | 14.29s | 3.5ms | 435.0 |
| D_A_lr1e-2 | 15.09s | 14.69s | 15.15s | 14.75s | 31.0ms | 15.15s | 3.9ms | 462.0 |
| D_F1000_lr3e-3 | 109.66s | 81.82s | 109.78s | 81.94s | 30.2ms | 109.79s | 4.1ms | 1000.0 |
| D_F1000_lr3e-3 | 32.10s | 31.76s | 32.16s | 31.82s | 29.0ms | 32.17s | 4.2ms | 1000.0 |
| D_F1000_lr3e-3 | 32.16s | 31.75s | 32.22s | 31.81s | 28.6ms | 32.23s | 4.4ms | 1000.0 |
| D_F1000_lr3e-3 | 32.35s | 31.80s | 32.41s | 31.86s | 28.8ms | 32.42s | 4.3ms | 1000.0 |
| D_F1000_lr3e-3 | 32.34s | 31.78s | 32.40s | 31.84s | 29.4ms | 32.41s | 4.2ms | 1000.0 |
| D_F1000_lr3e-3 | 32.19s | 31.76s | 32.25s | 31.82s | 29.3ms | 32.26s | 4.3ms | 1000.0 |
| D_F1000_lr3e-3 | 32.47s | 31.79s | 32.53s | 31.85s | 29.7ms | 32.54s | 4.0ms | 1000.0 |
| D_F1000_lr3e-3 | 32.18s | 31.79s | 32.26s | 31.87s | 53.4ms | 32.27s | 4.1ms | 1000.0 |
| D_F1000_lr3e-3 | 32.27s | 31.73s | 32.34s | 31.80s | 38.7ms | 32.35s | 4.1ms | 1000.0 |
| D_F1000_lr3e-3 | 32.14s | 31.73s | 32.21s | 31.80s | 35.5ms | 32.22s | 6.3ms | 1000.0 |
| D_F600_lr1e-2 | 92.93s | 62.26s | 93.05s | 62.38s | 33.9ms | 93.06s | 3.8ms | 600.0 |
| D_F600_lr1e-2 | 19.40s | 19.05s | 19.47s | 19.11s | 33.5ms | 19.48s | 4.3ms | 600.0 |
| D_F600_lr1e-2 | 19.43s | 19.02s | 19.49s | 19.09s | 33.6ms | 19.50s | 4.2ms | 600.0 |
| D_F600_lr1e-2 | 19.51s | 19.03s | 19.58s | 19.10s | 33.8ms | 19.59s | 6.7ms | 600.0 |
| D_F600_lr1e-2 | 19.53s | 19.07s | 19.61s | 19.16s | 50.8ms | 19.62s | 6.3ms | 600.0 |
| D_F600_lr1e-2 | 19.55s | 19.14s | 19.62s | 19.21s | 31.3ms | 19.63s | 4.0ms | 600.0 |
| D_F600_lr1e-2 | 19.50s | 19.05s | 19.57s | 19.12s | 31.4ms | 19.57s | 3.2ms | 600.0 |
| D_F600_lr1e-2 | 19.52s | 19.08s | 19.60s | 19.16s | 49.3ms | 19.61s | 4.0ms | 600.0 |
| D_F600_lr1e-2 | 19.43s | 19.05s | 19.49s | 19.11s | 31.6ms | 19.50s | 3.8ms | 600.0 |
| D_F600_lr1e-2 | 19.41s | 19.02s | 19.48s | 19.08s | 31.9ms | 19.49s | 6.7ms | 600.0 |
| D_baseline_lr1e-3_e2500 | 159.50s | 130.90s | 159.68s | 131.08s | 48.2ms | 159.69s | 5.3ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.10s | 80.63s | 81.20s | 80.73s | 43.4ms | 81.21s | 5.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.28s | 80.72s | 81.40s | 80.84s | 58.9ms | 81.40s | 4.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.14s | 80.70s | 81.25s | 80.81s | 49.9ms | 81.26s | 5.5ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.41s | 80.77s | 81.53s | 80.89s | 63.8ms | 81.54s | 9.1ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.15s | 80.69s | 81.27s | 80.80s | 52.4ms | 81.27s | 3.8ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.13s | 80.65s | 81.24s | 80.76s | 54.1ms | 81.25s | 4.9ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.09s | 80.64s | 81.18s | 80.73s | 32.3ms | 81.19s | 5.2ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.15s | 80.76s | 81.23s | 80.84s | 32.0ms | 81.24s | 5.1ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 81.11s | 80.70s | 81.20s | 80.79s | 32.5ms | 81.21s | 4.9ms | 2500.0 |
| S_A_lr1e-3 | 100.06s | 74.72s | 100.65s | 75.31s | 55.8ms | 100.66s | 4.1ms | 655.0 |
| S_A_lr1e-3 | 28.79s | 28.24s | 28.87s | 28.32s | 47.5ms | 28.88s | 4.1ms | 665.0 |
| S_A_lr1e-3 | 37.15s | 36.79s | 37.23s | 36.87s | 51.0ms | 37.24s | 4.4ms | 866.0 |
| S_A_lr1e-3 | 35.95s | 35.55s | 36.04s | 35.64s | 57.3ms | 36.05s | 4.3ms | 835.0 |
| S_A_lr1e-3 | 47.57s | 47.11s | 47.65s | 47.19s | 47.2ms | 47.66s | 4.5ms | 1104.0 |
| S_A_lr1e-3 | 45.90s | 45.48s | 45.99s | 45.57s | 52.9ms | 46.00s | 4.4ms | 1067.0 |
| S_A_lr1e-3 | 35.91s | 35.45s | 35.99s | 35.53s | 46.5ms | 36.00s | 4.4ms | 832.0 |
| S_A_lr1e-3 | 33.88s | 33.52s | 33.96s | 33.61s | 53.0ms | 33.97s | 4.2ms | 787.0 |
| S_A_lr1e-3 | 33.28s | 32.87s | 33.37s | 32.95s | 47.9ms | 33.37s | 4.5ms | 772.0 |
| S_A_lr1e-3 | 38.00s | 37.57s | 38.07s | 37.64s | 30.5ms | 38.07s | 4.2ms | 881.0 |
| S_F1000_lr1e-3 | 125.74s | 98.44s | 126.17s | 98.87s | 29.8ms | 126.17s | 4.2ms | 1000.0 |
| S_F1000_lr1e-3 | 42.93s | 42.55s | 43.01s | 42.64s | 49.3ms | 43.02s | 4.3ms | 1000.0 |
| S_F1000_lr1e-3 | 43.12s | 42.59s | 43.20s | 42.68s | 46.3ms | 43.21s | 4.2ms | 1000.0 |
| S_F1000_lr1e-3 | 43.17s | 42.72s | 43.25s | 42.81s | 48.3ms | 43.26s | 6.1ms | 1000.0 |
| S_F1000_lr1e-3 | 43.18s | 42.65s | 43.26s | 42.74s | 46.8ms | 43.27s | 4.2ms | 1000.0 |
| S_F1000_lr1e-3 | 43.12s | 42.65s | 43.21s | 42.74s | 50.9ms | 43.22s | 5.2ms | 1000.0 |
| S_F1000_lr1e-3 | 42.97s | 42.60s | 43.05s | 42.69s | 44.2ms | 43.06s | 4.5ms | 1000.0 |
| S_F1000_lr1e-3 | 43.17s | 42.72s | 43.26s | 42.81s | 48.6ms | 43.27s | 4.3ms | 1000.0 |
| S_F1000_lr1e-3 | 42.95s | 42.62s | 43.04s | 42.71s | 50.0ms | 43.05s | 6.2ms | 1000.0 |
| S_F1000_lr1e-3 | 43.09s | 42.66s | 43.17s | 42.74s | 43.9ms | 43.18s | 4.2ms | 1000.0 |
| S_baseline_lr1e-4_e2500 | 187.04s | 157.64s | 187.85s | 158.45s | 31.0ms | 187.86s | 5.0ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.24s | 106.78s | 107.32s | 106.85s | 31.3ms | 107.33s | 3.9ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.48s | 107.00s | 107.60s | 107.11s | 72.9ms | 107.61s | 4.9ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.07s | 106.73s | 107.16s | 106.81s | 45.9ms | 107.17s | 4.7ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.15s | 106.82s | 107.25s | 106.91s | 51.0ms | 107.25s | 4.9ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.30s | 106.74s | 107.39s | 106.84s | 47.3ms | 107.40s | 7.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.27s | 106.85s | 107.36s | 106.95s | 50.5ms | 107.37s | 4.9ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.16s | 106.78s | 107.23s | 106.85s | 29.4ms | 107.24s | 4.8ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.34s | 106.92s | 107.43s | 107.01s | 46.9ms | 107.44s | 4.8ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 107.20s | 106.86s | 107.27s | 106.93s | 29.3ms | 107.28s | 5.0ms | 2500.0 |

## Grouped Totals

| candidate | optimizer loop | non-loop training | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | DVF file write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D_A_lr1e-2 | 57.32s | 792.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 29.3ms |
| D_A_lr1e-2 | 12.15s | 5.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 31.4ms |
| D_A_lr1e-2 | 16.85s | 6.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 30.5ms |
| D_A_lr1e-2 | 17.43s | 6.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 31.5ms |
| D_A_lr1e-2 | 14.44s | 5.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 29.4ms |
| D_A_lr1e-2 | 15.39s | 5.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 29.9ms |
| D_A_lr1e-2 | 16.79s | 5.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 50.2ms |
| D_A_lr1e-2 | 13.89s | 5.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 32.6ms |
| D_A_lr1e-2 | 13.82s | 5.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 28.9ms |
| D_A_lr1e-2 | 14.68s | 5.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 31.0ms |
| D_F1000_lr3e-3 | 79.09s | 755.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 30.2ms |
| D_F1000_lr3e-3 | 31.75s | 8.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 29.0ms |
| D_F1000_lr3e-3 | 31.74s | 9.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 28.6ms |
| D_F1000_lr3e-3 | 31.79s | 10.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 28.8ms |
| D_F1000_lr3e-3 | 31.77s | 9.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 29.4ms |
| D_F1000_lr3e-3 | 31.75s | 8.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 29.3ms |
| D_F1000_lr3e-3 | 31.78s | 9.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 29.7ms |
| D_F1000_lr3e-3 | 31.78s | 9.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 53.4ms |
| D_F1000_lr3e-3 | 31.72s | 9.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 38.7ms |
| D_F1000_lr3e-3 | 31.72s | 8.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 35.5ms |
| D_F600_lr1e-2 | 61.08s | 800.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 33.9ms |
| D_F600_lr1e-2 | 19.04s | 6.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 33.5ms |
| D_F600_lr1e-2 | 19.01s | 6.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 33.6ms |
| D_F600_lr1e-2 | 19.02s | 7.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 33.8ms |
| D_F600_lr1e-2 | 19.06s | 6.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 50.8ms |
| D_F600_lr1e-2 | 19.14s | 6.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 31.3ms |
| D_F600_lr1e-2 | 19.05s | 6.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 31.4ms |
| D_F600_lr1e-2 | 19.07s | 6.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 49.3ms |
| D_F600_lr1e-2 | 19.04s | 6.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 31.6ms |
| D_F600_lr1e-2 | 19.01s | 6.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 31.9ms |
| D_baseline_lr1e-3_e2500 | 129.00s | 1.07s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 48.2ms |
| D_baseline_lr1e-3_e2500 | 80.60s | 26.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 43.4ms |
| D_baseline_lr1e-3_e2500 | 80.70s | 23.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 58.9ms |
| D_baseline_lr1e-3_e2500 | 80.67s | 23.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 49.9ms |
| D_baseline_lr1e-3_e2500 | 80.75s | 23.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 63.8ms |
| D_baseline_lr1e-3_e2500 | 80.67s | 23.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 52.4ms |
| D_baseline_lr1e-3_e2500 | 80.62s | 23.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 54.1ms |
| D_baseline_lr1e-3_e2500 | 80.62s | 23.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 32.3ms |
| D_baseline_lr1e-3_e2500 | 80.73s | 22.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 32.0ms |
| D_baseline_lr1e-3_e2500 | 80.67s | 23.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 32.5ms |
| S_A_lr1e-3 | 73.82s | 1.20s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 55.8ms |
| S_A_lr1e-3 | 28.23s | 11.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.2ms | 47.5ms |
| S_A_lr1e-3 | 36.78s | 13.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 51.0ms |
| S_A_lr1e-3 | 35.54s | 13.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 57.3ms |
| S_A_lr1e-3 | 47.09s | 15.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 47.2ms |
| S_A_lr1e-3 | 45.47s | 15.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 52.9ms |
| S_A_lr1e-3 | 35.43s | 13.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 46.5ms |
| S_A_lr1e-3 | 33.51s | 12.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 53.0ms |
| S_A_lr1e-3 | 32.86s | 13.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 47.9ms |
| S_A_lr1e-3 | 37.56s | 14.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 30.5ms |
| S_F1000_lr1e-3 | 96.81s | 1.08s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 29.8ms |
| S_F1000_lr1e-3 | 42.53s | 15.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 49.3ms |
| S_F1000_lr1e-3 | 42.58s | 14.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 46.3ms |
| S_F1000_lr1e-3 | 42.70s | 14.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 48.3ms |
| S_F1000_lr1e-3 | 42.63s | 14.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 46.8ms |
| S_F1000_lr1e-3 | 42.64s | 15.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 50.9ms |
| S_F1000_lr1e-3 | 42.58s | 17.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 44.2ms |
| S_F1000_lr1e-3 | 42.70s | 15.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 48.6ms |
| S_F1000_lr1e-3 | 42.60s | 15.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 50.0ms |
| S_F1000_lr1e-3 | 42.64s | 15.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 43.9ms |
| S_baseline_lr1e-4_e2500 | 155.75s | 1.61s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 31.0ms |
| S_baseline_lr1e-4_e2500 | 106.75s | 26.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 31.3ms |
| S_baseline_lr1e-4_e2500 | 106.97s | 26.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.6ms | 72.9ms |
| S_baseline_lr1e-4_e2500 | 106.70s | 26.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 45.9ms |
| S_baseline_lr1e-4_e2500 | 106.79s | 24.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 51.0ms |
| S_baseline_lr1e-4_e2500 | 106.72s | 25.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 47.3ms |
| S_baseline_lr1e-4_e2500 | 106.82s | 27.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 50.5ms |
| S_baseline_lr1e-4_e2500 | 106.75s | 24.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 29.4ms |
| S_baseline_lr1e-4_e2500 | 106.90s | 24.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 46.9ms |
| S_baseline_lr1e-4_e2500 | 106.83s | 29.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 29.3ms |

## Worked Numeric Examples

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 26.11
  + initial_dense_grid/small setup 0.30
  + stage 1 pre 0.73
  + stage 1 optimizer 57.32
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 84.47

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 84.47
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 84.59
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 12.15
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 12.59

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 12.59
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 12.65
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.56
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 16.85
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 17.42

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 17.42
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 17.48
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 17.43
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 17.86

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 17.86
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 17.93
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.51
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 14.44
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 14.95

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.95
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 15.01
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.32
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 15.39
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 15.72

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.72
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 15.77
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 16.79
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 17.23

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 17.23
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 17.31
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.47
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.89
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 14.36

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.36
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 14.42
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.82
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 14.22

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.22
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 14.29
```

```text
D_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 14.68
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 15.09

D_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.09
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 15.15
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 27.84
  + initial_dense_grid/small setup 2.02
  + stage 1 pre 0.69
  + stage 1 optimizer 79.09
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 109.66

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.66
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.09
  = 109.78
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.34
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.75
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.10

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.10
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.16
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.74
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.16

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.16
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.22
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.79
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.35

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.35
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.41
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.56
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.77
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.34

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.34
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.40
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.75
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.19

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.19
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.25
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.68
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.78
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.47

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.47
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.53
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.39
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.78
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.18

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.18
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.26
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.72
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.27

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.27
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.34
```

```text
D_F1000_lr3e-3 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 31.72
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 32.14

D_F1000_lr3e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 32.14
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 32.21
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 30.68
  + initial_dense_grid/small setup 0.43
  + stage 1 pre 0.74
  + stage 1 optimizer 61.08
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 92.93

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 92.93
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.09
  = 93.05
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.36
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.04
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.40

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.40
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.47
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.01
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.43

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.43
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.49
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.48
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.02
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.51

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.51
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.58
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.06
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.53

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.53
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.61
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.14
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.55

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.55
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.62
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.05
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.50

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.50
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.57
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.07
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.52

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.52
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.60
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.04
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.43

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.43
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.49
```

```text
D_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 19.01
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 19.41

D_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 19.41
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 19.48
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 28.60
  + initial_dense_grid/small setup 0.90
  + stage 1 pre 0.98
  + stage 1 optimizer 129.00
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 159.50

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 159.50
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.13
  = 159.68
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.47
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.60
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.10

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.10
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.20
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.70
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.28

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.28
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.40
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.67
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.14

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.14
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.25
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.64
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.75
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.41

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.41
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.53
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.67
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.15

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.15
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.27
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.48
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.62
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.13

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.13
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.24
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.62
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.09

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.09
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.18
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.39
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.73
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.15

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.15
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 81.23
```

```text
D_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 80.67
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.02
  = 81.11

D_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 81.11
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 81.20
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 25.34
  + initial_dense_grid/small setup 0.20
  + stage 1 pre 0.69
  + stage 1 optimizer 73.82
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 100.06

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 100.06
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.53
  = 100.65
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.56
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 28.23
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.00
  = 28.79

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 28.79
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 28.87
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.35
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 36.78
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 37.15

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 37.15
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 37.23
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 35.54
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 35.95

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 35.95
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 36.04
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 47.09
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 47.57

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 47.57
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 47.65
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 45.47
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 45.90

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 45.90
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 45.99
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 35.43
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 35.91

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 35.91
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 35.99
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.35
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 33.51
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 33.88

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 33.88
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 33.96
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 32.86
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 33.28

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 33.28
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 33.37
```

```text
S_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 37.56
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 38.00

S_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 38.00
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 38.07
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 27.30
  + initial_dense_grid/small setup 0.91
  + stage 1 pre 0.71
  + stage 1 optimizer 96.81
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 125.74

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 125.74
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.40
  = 126.17
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.53
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 42.93

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 42.93
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.01
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.52
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.58
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.12

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.12
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.20
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.70
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.17

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.17
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.25
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.53
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.63
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.18

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.18
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.26
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.47
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.64
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.12

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.12
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.21
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.36
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.58
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 42.97

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 42.97
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.05
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.70
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.17

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.17
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.26
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.33
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.60
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 42.95

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 42.95
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.04
```

```text
S_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 42.64
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.01
  = 43.09

S_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 43.09
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 43.17
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 29.40
  + initial_dense_grid/small setup 1.02
  + stage 1 pre 0.86
  + stage 1 optimizer 155.75
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 187.04

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 187.04
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.78
  = 187.85
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.47
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.75
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.24

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.24
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.32
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.48
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.97
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.48

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.48
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.60
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.35
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.70
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.07

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.07
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.16
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.34
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.79
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.15

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.15
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.25
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.72
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.30

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.30
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.39
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.82
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.27

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.27
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 107.36
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.75
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.16

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.16
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.23
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.90
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.34

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.34
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 107.43
```

```text
S_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.34
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.83
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.20

S_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.20
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 107.27
```

## Previous Conservative Timing Comparison

Positive values mean the new production path is faster than the previous conservative DVF-only timing.
Endpoint rows are the primary comparison. Component rows can shift because skipping earlier dense expansions also changes where CUDA warmup/autotune cost appears.

| candidate | metric | previous | production | previous - production |
|---|---|---:|---:|---:|
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 159.50s | -157.56s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 130.90s | -129.36s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 159.68s | -157.65s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 131.08s | -129.45s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 1.07s | -1.06s |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 129.00s | -127.46s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 159.69s | -157.65s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.10s | -79.16s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.63s | -79.09s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.20s | -79.17s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.73s | -79.10s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 26.0ms | -21.4ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.60s | -79.06s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.21s | -79.17s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.28s | -79.34s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.72s | -79.18s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.40s | -79.37s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.84s | -79.22s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.5ms | -18.9ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.70s | -79.16s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.40s | -79.37s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.14s | -79.20s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.70s | -79.16s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.25s | -79.22s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.81s | -79.18s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.8ms | -19.2ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.67s | -79.14s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.26s | -79.22s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.41s | -79.47s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.77s | -79.23s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.53s | -79.50s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.89s | -79.27s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.0ms | -18.4ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.75s | -79.21s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.54s | -79.51s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.15s | -79.21s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.69s | -79.15s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.27s | -79.24s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.80s | -79.18s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.5ms | -18.9ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.67s | -79.13s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.27s | -79.24s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.13s | -79.19s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.65s | -79.10s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.24s | -79.22s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.76s | -79.13s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.9ms | -19.2ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.62s | -79.08s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.25s | -79.22s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.09s | -79.15s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.64s | -79.10s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.18s | -79.15s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.73s | -79.10s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.5ms | -18.9ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.62s | -79.08s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.19s | -79.15s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.15s | -79.21s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.76s | -79.22s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.23s | -79.21s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.84s | -79.22s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 22.9ms | -18.3ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.73s | -79.20s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.24s | -79.21s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 1.94s | 81.11s | -79.17s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 1.54s | 80.70s | -79.16s |
| D_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 2.03s | 81.20s | -79.17s |
| D_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 1.63s | 80.79s | -79.16s |
| D_baseline_lr1e-3_e2500 | total_non_loop_training | 4.6ms | 23.4ms | -18.8ms |
| D_baseline_lr1e-3_e2500 | total_optimizer_loop | 1.54s | 80.67s | -79.14s |
| D_baseline_lr1e-3_e2500 | external_wall | 2.04s | 81.21s | -79.17s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 187.04s | -186.49s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 157.64s | -157.52s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 187.85s | -187.22s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 158.45s | -158.25s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 1.61s | -1.60s |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 155.75s | -155.63s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 187.86s | -187.22s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.24s | -106.70s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.78s | -106.66s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.32s | -106.69s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.85s | -106.65s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 26.4ms | -18.2ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.75s | -106.64s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.33s | -106.69s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.48s | -106.93s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 107.00s | -106.87s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.60s | -106.97s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 107.11s | -106.91s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 26.4ms | -18.2ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.97s | -106.86s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.61s | -106.97s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.07s | -106.53s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.73s | -106.60s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.16s | -106.54s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.81s | -106.61s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 26.1ms | -17.9ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.70s | -106.59s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.17s | -106.54s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.15s | -106.61s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.82s | -106.70s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.25s | -106.62s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.91s | -106.71s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 24.0ms | -15.8ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.79s | -106.68s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.25s | -106.62s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.30s | -106.75s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.74s | -106.62s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.39s | -106.76s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.84s | -106.63s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 25.7ms | -17.5ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.72s | -106.60s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.40s | -106.77s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.27s | -106.72s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.85s | -106.73s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.36s | -106.74s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.95s | -106.75s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 27.7ms | -19.6ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.82s | -106.71s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.37s | -106.74s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.16s | -106.61s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.78s | -106.65s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.23s | -106.61s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.85s | -106.65s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 24.8ms | -16.7ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.75s | -106.64s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.24s | -106.61s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.34s | -106.79s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.92s | -106.80s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.43s | -106.80s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 107.01s | -106.81s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 24.5ms | -16.3ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.90s | -106.79s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.44s | -106.81s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 546.4ms | 107.20s | -106.65s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 122.8ms | 106.86s | -106.73s |
| S_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 625.2ms | 107.27s | -106.65s |
| S_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 201.6ms | 106.93s | -106.73s |
| S_baseline_lr1e-4_e2500 | total_non_loop_training | 8.2ms | 29.3ms | -21.1ms |
| S_baseline_lr1e-4_e2500 | total_optimizer_loop | 113.5ms | 106.83s | -106.71s |
| S_baseline_lr1e-4_e2500 | external_wall | 632.6ms | 107.28s | -106.65s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 109.66s | -108.45s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 81.82s | -81.05s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 109.78s | -108.50s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 81.94s | -81.10s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 755.1ms | -750.4ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 79.09s | -78.33s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 109.79s | -108.50s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.10s | -30.90s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.76s | -31.00s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.16s | -30.89s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.82s | -30.98s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 8.9ms | -4.2ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.75s | -30.99s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.17s | -30.89s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.16s | -30.95s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.75s | -30.99s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.22s | -30.94s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.81s | -30.98s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 9.0ms | -4.3ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.74s | -30.98s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.23s | -30.94s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.35s | -31.15s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.80s | -31.04s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.41s | -31.13s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.86s | -31.03s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 10.3ms | -5.6ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.79s | -31.03s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.42s | -31.13s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.34s | -31.13s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.78s | -31.02s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.40s | -31.12s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.84s | -31.01s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 9.3ms | -4.6ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.77s | -31.02s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.41s | -31.12s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.19s | -30.99s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.76s | -30.99s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.25s | -30.98s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.82s | -30.98s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 8.8ms | -4.1ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.75s | -30.99s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.26s | -30.98s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.47s | -31.26s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.79s | -31.02s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.53s | -31.25s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.85s | -31.01s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 9.2ms | -4.5ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.78s | -31.02s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.54s | -31.25s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.18s | -30.97s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.79s | -31.02s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.26s | -30.99s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.87s | -31.04s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 9.2ms | -4.4ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.78s | -31.02s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.27s | -30.99s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.27s | -31.07s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.73s | -30.96s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.34s | -31.07s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.80s | -30.96s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 9.6ms | -4.9ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.72s | -30.96s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.35s | -31.07s |
| D_F1000_lr3e-3 | full_script_to_dvf_in_memory | 1.21s | 32.14s | -30.94s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_in_memory | 763.9ms | 31.73s | -30.97s |
| D_F1000_lr3e-3 | full_script_to_dvf_file | 1.28s | 32.21s | -30.93s |
| D_F1000_lr3e-3 | loaded_images_to_dvf_file | 834.5ms | 31.80s | -30.96s |
| D_F1000_lr3e-3 | total_non_loop_training | 4.7ms | 8.9ms | -4.1ms |
| D_F1000_lr3e-3 | total_optimizer_loop | 759.2ms | 31.72s | -30.96s |
| D_F1000_lr3e-3 | external_wall | 1.28s | 32.22s | -30.93s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 92.93s | -91.47s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 62.26s | -61.20s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 93.05s | -91.52s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 62.38s | -61.25s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 800.2ms | -795.8ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 61.08s | -60.03s |
| D_F600_lr1e-2 | external_wall | 1.54s | 93.06s | -91.52s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.40s | -17.95s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.05s | -17.99s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.47s | -17.94s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.11s | -17.99s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.9ms | -2.4ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.04s | -17.99s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.48s | -17.94s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.43s | -17.97s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.02s | -17.97s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.49s | -17.96s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.09s | -17.96s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.6ms | -2.2ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.01s | -17.96s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.50s | -17.97s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.51s | -18.06s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.03s | -17.98s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.58s | -18.05s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.10s | -17.98s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 7.1ms | -2.7ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.02s | -17.98s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.59s | -18.06s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.53s | -18.07s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.07s | -18.02s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.61s | -18.08s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.16s | -18.03s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.3ms | -1.8ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.06s | -18.01s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.62s | -18.09s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.55s | -18.09s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.14s | -18.09s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.62s | -18.09s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.21s | -18.08s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.4ms | -2.0ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.14s | -18.09s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.63s | -18.09s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.50s | -18.04s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.05s | -18.00s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.57s | -18.03s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.12s | -17.99s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.5ms | -2.1ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.05s | -18.00s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.57s | -18.04s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.52s | -18.06s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.08s | -18.03s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.60s | -18.07s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.16s | -18.03s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.2ms | -1.7ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.07s | -18.02s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.61s | -18.08s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.43s | -17.97s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.05s | -17.99s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.49s | -17.96s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.11s | -17.99s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.7ms | -2.3ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.04s | -17.99s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.50s | -17.97s |
| D_F600_lr1e-2 | full_script_to_dvf_in_memory | 1.46s | 19.41s | -17.95s |
| D_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 1.05s | 19.02s | -17.96s |
| D_F600_lr1e-2 | full_script_to_dvf_file | 1.53s | 19.48s | -17.95s |
| D_F600_lr1e-2 | loaded_images_to_dvf_file | 1.13s | 19.08s | -17.96s |
| D_F600_lr1e-2 | total_non_loop_training | 4.4ms | 6.7ms | -2.3ms |
| D_F600_lr1e-2 | total_optimizer_loop | 1.05s | 19.01s | -17.96s |
| D_F600_lr1e-2 | external_wall | 1.54s | 19.49s | -17.95s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 84.47s | -83.72s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 58.36s | -57.99s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 84.59s | -83.75s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 58.47s | -58.02s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 792.7ms | -788.1ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 57.32s | -56.96s |
| D_A_lr1e-2 | external_wall | 846.0ms | 84.59s | -83.75s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 12.59s | -11.83s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 12.16s | -11.79s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 12.65s | -11.81s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 12.22s | -11.77s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.6ms | -0.9ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 12.15s | -11.79s |
| D_A_lr1e-2 | external_wall | 846.0ms | 12.66s | -11.81s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 17.42s | -16.66s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 16.86s | -16.49s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 17.48s | -16.64s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 16.92s | -16.47s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 6.0ms | -1.3ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 16.85s | -16.49s |
| D_A_lr1e-2 | external_wall | 846.0ms | 17.49s | -16.64s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 17.86s | -17.10s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 17.44s | -17.07s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 17.93s | -17.09s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 17.50s | -17.05s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 6.6ms | -2.0ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 17.43s | -17.07s |
| D_A_lr1e-2 | external_wall | 846.0ms | 17.93s | -17.09s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 14.95s | -14.20s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 14.44s | -14.07s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 15.01s | -14.17s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 14.50s | -14.05s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.2ms | -0.5ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 14.44s | -14.07s |
| D_A_lr1e-2 | external_wall | 846.0ms | 15.01s | -14.17s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 15.72s | -14.96s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 15.40s | -15.03s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 15.77s | -14.93s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 15.45s | -15.00s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.4ms | -0.8ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 15.39s | -15.03s |
| D_A_lr1e-2 | external_wall | 846.0ms | 15.78s | -14.93s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 17.23s | -16.48s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 16.80s | -16.43s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 17.31s | -16.47s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 16.87s | -16.42s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.9ms | -1.3ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 16.79s | -16.43s |
| D_A_lr1e-2 | external_wall | 846.0ms | 17.32s | -16.47s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 14.36s | -13.60s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 13.89s | -13.53s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 14.42s | -13.58s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 13.95s | -13.50s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.7ms | -1.1ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 13.89s | -13.52s |
| D_A_lr1e-2 | external_wall | 846.0ms | 14.43s | -13.58s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 14.22s | -13.47s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 13.83s | -13.46s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 14.29s | -13.45s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 13.89s | -13.44s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.0ms | -0.4ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 13.82s | -13.46s |
| D_A_lr1e-2 | external_wall | 846.0ms | 14.29s | -13.45s |
| D_A_lr1e-2 | full_script_to_dvf_in_memory | 757.1ms | 15.09s | -14.33s |
| D_A_lr1e-2 | loaded_images_to_dvf_in_memory | 368.3ms | 14.69s | -14.32s |
| D_A_lr1e-2 | full_script_to_dvf_file | 838.6ms | 15.15s | -14.31s |
| D_A_lr1e-2 | loaded_images_to_dvf_file | 449.8ms | 14.75s | -14.30s |
| D_A_lr1e-2 | total_non_loop_training | 4.6ms | 5.8ms | -1.2ms |
| D_A_lr1e-2 | total_optimizer_loop | 363.6ms | 14.68s | -14.32s |
| D_A_lr1e-2 | external_wall | 846.0ms | 15.15s | -14.31s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 125.74s | -125.12s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 98.44s | -98.16s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 126.17s | -125.27s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 98.87s | -98.30s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 1.08s | -1.07s |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 96.81s | -96.53s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 126.17s | -125.27s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 42.93s | -42.31s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.55s | -42.26s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.01s | -42.12s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.64s | -42.07s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 15.0ms | -7.1ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.53s | -42.26s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.02s | -42.12s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.12s | -42.50s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.59s | -42.31s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.20s | -42.31s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.68s | -42.11s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 14.2ms | -6.3ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.58s | -42.30s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.21s | -42.31s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.17s | -42.55s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.72s | -42.44s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.25s | -42.36s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.81s | -42.24s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 14.8ms | -6.9ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.70s | -42.43s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.26s | -42.36s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.18s | -42.56s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.65s | -42.36s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.26s | -42.37s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.74s | -42.17s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 14.7ms | -6.7ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.63s | -42.36s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.27s | -42.37s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.12s | -42.51s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.65s | -42.37s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.21s | -42.31s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.74s | -42.18s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 15.6ms | -7.6ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.64s | -42.36s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.22s | -42.32s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 42.97s | -42.35s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.60s | -42.32s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.05s | -42.15s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.69s | -42.12s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 17.0ms | -9.1ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.58s | -42.31s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.06s | -42.15s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.17s | -42.56s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.72s | -42.44s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.26s | -42.36s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.81s | -42.24s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 15.3ms | -7.3ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.70s | -42.43s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.27s | -42.37s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 42.95s | -42.33s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.62s | -42.34s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.04s | -42.14s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.71s | -42.15s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 15.1ms | -7.2ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.60s | -42.33s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.05s | -42.14s |
| S_F1000_lr1e-3 | full_script_to_dvf_in_memory | 615.6ms | 43.09s | -42.47s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 284.9ms | 42.66s | -42.38s |
| S_F1000_lr1e-3 | full_script_to_dvf_file | 896.1ms | 43.17s | -42.27s |
| S_F1000_lr1e-3 | loaded_images_to_dvf_file | 565.3ms | 42.74s | -42.18s |
| S_F1000_lr1e-3 | total_non_loop_training | 7.9ms | 15.7ms | -7.8ms |
| S_F1000_lr1e-3 | total_optimizer_loop | 276.1ms | 42.64s | -42.37s |
| S_F1000_lr1e-3 | external_wall | 902.4ms | 43.18s | -42.27s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 100.06s | -99.05s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 74.72s | -74.21s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 100.65s | -99.55s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 75.31s | -74.71s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 1.20s | -1.19s |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 73.82s | -73.32s |
| S_A_lr1e-3 | external_wall | 1.10s | 100.66s | -99.55s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 28.79s | -27.79s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 28.24s | -27.73s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 28.87s | -27.78s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 28.32s | -27.73s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 11.9ms | -3.4ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 28.23s | -27.73s |
| S_A_lr1e-3 | external_wall | 1.10s | 28.88s | -27.78s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 37.15s | -36.14s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 36.79s | -36.28s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 37.23s | -36.14s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 36.87s | -36.28s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 13.2ms | -4.7ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 36.78s | -36.28s |
| S_A_lr1e-3 | external_wall | 1.10s | 37.24s | -36.14s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 35.95s | -34.94s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 35.55s | -35.04s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 36.04s | -34.95s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 35.64s | -35.05s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 13.6ms | -5.0ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 35.54s | -35.04s |
| S_A_lr1e-3 | external_wall | 1.10s | 36.05s | -34.94s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 47.57s | -46.56s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 47.11s | -46.60s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 47.65s | -46.56s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 47.19s | -46.60s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 15.7ms | -7.2ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 47.09s | -46.59s |
| S_A_lr1e-3 | external_wall | 1.10s | 47.66s | -46.56s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 45.90s | -44.90s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 45.48s | -44.98s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 45.99s | -44.90s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 45.57s | -44.98s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 15.6ms | -7.1ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 45.47s | -44.97s |
| S_A_lr1e-3 | external_wall | 1.10s | 46.00s | -44.90s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 35.91s | -34.90s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 35.45s | -34.94s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 35.99s | -34.90s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 35.53s | -34.93s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 13.5ms | -5.0ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 35.43s | -34.93s |
| S_A_lr1e-3 | external_wall | 1.10s | 36.00s | -34.90s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 33.88s | -32.87s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 33.52s | -33.02s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 33.96s | -32.87s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 33.61s | -33.02s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 12.7ms | -4.1ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 33.51s | -33.01s |
| S_A_lr1e-3 | external_wall | 1.10s | 33.97s | -32.87s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 33.28s | -32.28s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 32.87s | -32.36s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 33.37s | -32.27s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 32.95s | -32.36s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 13.2ms | -4.7ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 32.86s | -32.36s |
| S_A_lr1e-3 | external_wall | 1.10s | 33.37s | -32.27s |
| S_A_lr1e-3 | full_script_to_dvf_in_memory | 1.01s | 38.00s | -36.99s |
| S_A_lr1e-3 | loaded_images_to_dvf_in_memory | 508.6ms | 37.57s | -37.06s |
| S_A_lr1e-3 | full_script_to_dvf_file | 1.09s | 38.07s | -36.97s |
| S_A_lr1e-3 | loaded_images_to_dvf_file | 592.6ms | 37.64s | -37.05s |
| S_A_lr1e-3 | total_non_loop_training | 8.5ms | 14.4ms | -5.9ms |
| S_A_lr1e-3 | total_optimizer_loop | 498.9ms | 37.56s | -37.06s |
| S_A_lr1e-3 | external_wall | 1.10s | 38.07s | -36.97s |

## Stage Breakdown

| candidate | stage | pre | optimizer | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | post-to-DVF | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_A_lr1e-2 | 1 | 731.5ms | 57.32s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 510.0 |
| D_A_lr1e-2 | 1 | 1.2ms | 12.15s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 381.0 |
| D_A_lr1e-2 | 1 | 0.6ms | 16.85s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 530.0 |
| D_A_lr1e-2 | 1 | 0.8ms | 17.43s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 547.0 |
| D_A_lr1e-2 | 1 | 0.6ms | 14.44s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 454.0 |
| D_A_lr1e-2 | 1 | 0.6ms | 15.39s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 485.0 |
| D_A_lr1e-2 | 1 | 0.7ms | 16.79s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 528.0 |
| D_A_lr1e-2 | 1 | 0.7ms | 13.89s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 437.0 |
| D_A_lr1e-2 | 1 | 0.6ms | 13.82s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 435.0 |
| D_A_lr1e-2 | 1 | 0.6ms | 14.68s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 462.0 |
| D_F1000_lr3e-3 | 1 | 691.0ms | 79.09s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.6ms | 31.75s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.8ms | 31.74s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 1.1ms | 31.79s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.8ms | 31.77s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.7ms | 31.75s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.6ms | 31.78s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.6ms | 31.78s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.8ms | 31.72s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 1000.0 |
| D_F1000_lr3e-3 | 1 | 0.7ms | 31.72s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 1000.0 |
| D_F600_lr1e-2 | 1 | 743.1ms | 61.08s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.7ms | 19.04s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.7ms | 19.01s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.8ms | 19.02s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.7ms | 19.06s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.6ms | 19.14s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.8ms | 19.05s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.6ms | 19.07s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.6ms | 19.04s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D_F600_lr1e-2 | 1 | 0.8ms | 19.01s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D_baseline_lr1e-3_e2500 | 1 | 980.7ms | 129.00s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 1.9ms | 80.60s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 1.0ms | 80.70s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.9ms | 80.67s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.9ms | 80.75s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.9ms | 80.67s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.8ms | 80.62s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.8ms | 80.62s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.7ms | 80.73s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 2500.0 |
| D_baseline_lr1e-3_e2500 | 1 | 0.9ms | 80.67s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 2500.0 |
| S_A_lr1e-3 | 1 | 688.3ms | 73.82s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 655.0 |
| S_A_lr1e-3 | 1 | 2.0ms | 28.23s | 0.00s | 0.00s | 0.00s | 0.00s | 5.2ms | 5.2ms | 665.0 |
| S_A_lr1e-3 | 1 | 1.9ms | 36.78s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 866.0 |
| S_A_lr1e-3 | 1 | 2.1ms | 35.54s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 835.0 |
| S_A_lr1e-3 | 1 | 2.0ms | 47.09s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1104.0 |
| S_A_lr1e-3 | 1 | 2.0ms | 45.47s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1067.0 |
| S_A_lr1e-3 | 1 | 2.1ms | 35.43s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 832.0 |
| S_A_lr1e-3 | 1 | 2.0ms | 33.51s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 787.0 |
| S_A_lr1e-3 | 1 | 1.9ms | 32.86s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 772.0 |
| S_A_lr1e-3 | 1 | 1.9ms | 37.56s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 881.0 |
| S_F1000_lr1e-3 | 1 | 710.4ms | 96.81s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.3ms | 42.53s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.0ms | 42.58s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.4ms | 42.70s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 1.9ms | 42.63s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.0ms | 42.64s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.3ms | 42.58s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.3ms | 42.70s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 2.0ms | 42.60s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_F1000_lr1e-3 | 1 | 1.9ms | 42.64s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 1000.0 |
| S_baseline_lr1e-4_e2500 | 1 | 855.0ms | 155.75s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.1ms | 106.75s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.0ms | 106.97s | 0.00s | 0.00s | 0.00s | 0.00s | 5.6ms | 5.6ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 1.9ms | 106.70s | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 5.5ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 1.9ms | 106.79s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.1ms | 106.72s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.3ms | 106.82s | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 5.5ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 1.9ms | 106.75s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.0ms | 106.90s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| S_baseline_lr1e-4_e2500 | 1 | 2.2ms | 106.83s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |

## Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_first10.csv`
