# OASIS DVF-Only Runtime Timing

- Phase: `first10`
- Output root: `<container-data-root>/inrorspline/runs/oasis_cps2_single_stage_persistent_dvf_timing_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-06_oasis_cps2_single_stage_runtime_sweep`
- Completed command rows: 90
- Timing JSON rows: 90/90
- Dense DVF files present: 90/90
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
| D2_A_lr1e-2 | 109.26s | 79.68s | 109.98s | 80.39s | 90.2ms | 109.99s | 5.2ms | 331.0 |
| D2_A_lr1e-2 | 15.74s | 12.67s | 15.86s | 12.79s | 71.1ms | 15.88s | 8.0ms | 350.0 |
| D2_A_lr1e-2 | 12.39s | 11.56s | 12.51s | 11.68s | 69.3ms | 12.52s | 4.6ms | 325.0 |
| D2_A_lr1e-2 | 18.16s | 17.51s | 18.28s | 17.64s | 72.1ms | 18.30s | 5.2ms | 491.0 |
| D2_A_lr1e-2 | 12.98s | 12.39s | 13.07s | 12.48s | 48.0ms | 13.07s | 4.5ms | 355.0 |
| D2_A_lr1e-2 | 14.27s | 13.55s | 14.42s | 13.70s | 96.6ms | 14.43s | 4.8ms | 399.0 |
| D2_A_lr1e-2 | 13.52s | 12.84s | 13.65s | 12.97s | 68.8ms | 13.66s | 4.0ms | 372.0 |
| D2_A_lr1e-2 | 15.17s | 14.52s | 15.32s | 14.67s | 69.6ms | 15.33s | 4.1ms | 410.0 |
| D2_A_lr1e-2 | 12.26s | 11.42s | 12.44s | 11.60s | 96.1ms | 12.46s | 4.8ms | 314.0 |
| D2_A_lr1e-2 | 16.20s | 15.65s | 16.38s | 15.83s | 98.0ms | 16.39s | 4.7ms | 450.0 |
| D2_A_lr3e-2 | 89.53s | 76.50s | 89.99s | 76.96s | 90.6ms | 90.00s | 4.6ms | 200.0 |
| D2_A_lr3e-2 | 7.61s | 7.00s | 7.74s | 7.14s | 65.1ms | 7.75s | 4.5ms | 202.0 |
| D2_A_lr3e-2 | 8.01s | 7.18s | 8.18s | 7.34s | 86.1ms | 8.19s | 4.3ms | 200.0 |
| D2_A_lr3e-2 | 8.95s | 7.71s | 9.12s | 7.87s | 90.9ms | 9.14s | 4.5ms | 207.0 |
| D2_A_lr3e-2 | 9.28s | 8.44s | 9.46s | 8.62s | 92.9ms | 9.47s | 5.3ms | 239.0 |
| D2_A_lr3e-2 | 8.19s | 7.40s | 8.35s | 7.56s | 88.5ms | 8.37s | 4.6ms | 206.0 |
| D2_A_lr3e-2 | 9.20s | 8.34s | 9.33s | 8.47s | 72.6ms | 9.34s | 5.2ms | 228.0 |
| D2_A_lr3e-2 | 9.37s | 8.60s | 9.49s | 8.72s | 65.8ms | 9.50s | 4.9ms | 242.0 |
| D2_A_lr3e-2 | 8.43s | 7.44s | 8.58s | 7.59s | 78.3ms | 8.59s | 5.5ms | 205.0 |
| D2_A_lr3e-2 | 8.84s | 8.12s | 8.99s | 8.27s | 76.9ms | 9.00s | 6.0ms | 214.0 |
| D2_F400_lr3e-2 | 100.25s | 83.23s | 100.68s | 83.67s | 99.3ms | 100.69s | 4.7ms | 400.0 |
| D2_F400_lr3e-2 | 14.42s | 13.69s | 14.53s | 13.80s | 53.9ms | 14.53s | 4.4ms | 400.0 |
| D2_F400_lr3e-2 | 14.17s | 13.54s | 14.29s | 13.66s | 56.5ms | 14.29s | 4.8ms | 400.0 |
| D2_F400_lr3e-2 | 14.35s | 13.71s | 14.52s | 13.88s | 88.6ms | 14.53s | 5.1ms | 400.0 |
| D2_F400_lr3e-2 | 14.84s | 14.11s | 15.00s | 14.26s | 81.9ms | 15.01s | 5.1ms | 400.0 |
| D2_F400_lr3e-2 | 14.77s | 13.97s | 14.92s | 14.12s | 89.0ms | 14.94s | 5.3ms | 400.0 |
| D2_F400_lr3e-2 | 15.56s | 14.80s | 15.70s | 14.94s | 75.9ms | 15.72s | 6.7ms | 400.0 |
| D2_F400_lr3e-2 | 15.01s | 14.45s | 15.15s | 14.59s | 79.4ms | 15.16s | 5.0ms | 400.0 |
| D2_F400_lr3e-2 | 15.52s | 13.98s | 15.66s | 14.12s | 77.6ms | 15.68s | 5.0ms | 400.0 |
| D2_F400_lr3e-2 | 18.66s | 14.40s | 18.80s | 14.54s | 72.5ms | 18.82s | 5.2ms | 400.0 |
| D2_F600_lr1e-2 | 103.35s | 86.98s | 103.71s | 87.34s | 44.7ms | 103.72s | 4.2ms | 600.0 |
| D2_F600_lr1e-2 | 21.02s | 20.34s | 21.10s | 20.42s | 45.6ms | 21.11s | 4.2ms | 600.0 |
| D2_F600_lr1e-2 | 20.98s | 20.33s | 21.07s | 20.42s | 46.0ms | 21.08s | 4.1ms | 600.0 |
| D2_F600_lr1e-2 | 21.42s | 20.84s | 21.55s | 20.98s | 82.3ms | 21.56s | 4.5ms | 600.0 |
| D2_F600_lr1e-2 | 21.49s | 20.86s | 21.62s | 21.00s | 76.7ms | 21.63s | 5.3ms | 600.0 |
| D2_F600_lr1e-2 | 21.34s | 20.77s | 21.49s | 20.92s | 80.3ms | 21.50s | 4.8ms | 600.0 |
| D2_F600_lr1e-2 | 20.98s | 20.29s | 21.12s | 20.44s | 81.0ms | 21.14s | 7.3ms | 600.0 |
| D2_F600_lr1e-2 | 21.01s | 20.40s | 21.13s | 20.51s | 56.1ms | 21.14s | 4.3ms | 600.0 |
| D2_F600_lr1e-2 | 21.54s | 20.82s | 21.66s | 20.94s | 56.6ms | 21.67s | 4.2ms | 600.0 |
| D2_F600_lr1e-2 | 21.01s | 20.36s | 21.13s | 20.48s | 56.0ms | 21.14s | 4.5ms | 600.0 |
| D2_baseline_lr1e-3_e2500 | 172.57s | 156.42s | 173.02s | 156.86s | 79.7ms | 173.04s | 6.7ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 86.41s | 85.74s | 86.53s | 85.86s | 54.0ms | 86.55s | 6.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 84.43s | 83.81s | 84.59s | 83.97s | 78.7ms | 84.63s | 9.0ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 85.95s | 85.27s | 86.08s | 85.39s | 54.6ms | 86.23s | 128.5ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 86.17s | 85.48s | 86.29s | 85.60s | 53.3ms | 86.33s | 12.4ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 85.22s | 84.51s | 85.33s | 84.62s | 51.7ms | 85.86s | 10.3ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 85.87s | 85.19s | 86.01s | 85.34s | 72.8ms | 86.04s | 6.9ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 85.93s | 85.38s | 86.09s | 85.55s | 80.5ms | 86.12s | 6.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 86.19s | 85.40s | 86.35s | 85.56s | 70.3ms | 86.38s | 6.7ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 85.67s | 85.17s | 85.81s | 85.31s | 76.4ms | 85.84s | 9.4ms | 2500.0 |
| S2_A_lr1e-3 | 220.90s | 185.28s | 221.06s | 185.44s | 41.3ms | 221.07s | 4.6ms | 1018.0 |
| S2_A_lr1e-3 | 86.75s | 86.21s | 86.85s | 86.31s | 43.3ms | 87.68s | 5.0ms | 795.0 |
| S2_A_lr1e-3 | 157.41s | 156.67s | 157.51s | 156.77s | 46.9ms | 157.52s | 4.8ms | 1438.0 |
| S2_A_lr1e-3 | 106.78s | 106.34s | 106.86s | 106.43s | 42.1ms | 106.87s | 4.1ms | 980.0 |
| S2_A_lr1e-3 | 112.81s | 112.27s | 112.90s | 112.35s | 42.0ms | 112.91s | 5.3ms | 1033.0 |
| S2_A_lr1e-3 | 123.43s | 122.79s | 123.51s | 122.87s | 39.7ms | 123.52s | 4.7ms | 1133.0 |
| S2_A_lr1e-3 | 126.86s | 126.21s | 126.95s | 126.30s | 42.5ms | 126.96s | 4.6ms | 1161.0 |
| S2_A_lr1e-3 | 99.86s | 99.21s | 99.97s | 99.32s | 67.9ms | 99.98s | 4.9ms | 910.0 |
| S2_A_lr1e-3 | 84.23s | 83.79s | 84.31s | 83.87s | 42.3ms | 84.32s | 4.0ms | 772.0 |
| S2_A_lr1e-3 | 129.22s | 128.17s | 129.30s | 128.26s | 41.7ms | 129.31s | 4.7ms | 1177.0 |
| S2_F1000_lr1e-3 | 186.33s | 161.17s | 186.49s | 161.33s | 34.9ms | 186.50s | 4.3ms | 1000.0 |
| S2_F1000_lr1e-3 | 108.08s | 107.67s | 108.16s | 107.75s | 34.3ms | 108.17s | 4.3ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.82s | 109.36s | 109.92s | 109.46s | 45.3ms | 109.93s | 4.6ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.48s | 108.62s | 109.63s | 108.77s | 85.0ms | 109.64s | 5.0ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.67s | 108.89s | 109.77s | 108.99s | 45.6ms | 109.78s | 4.5ms | 1000.0 |
| S2_F1000_lr1e-3 | 108.63s | 108.10s | 108.73s | 108.20s | 44.2ms | 108.74s | 4.5ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.64s | 109.00s | 109.78s | 109.13s | 62.3ms | 109.79s | 5.1ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.11s | 108.34s | 109.21s | 108.43s | 43.8ms | 109.22s | 4.5ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.61s | 109.04s | 109.74s | 109.17s | 67.1ms | 109.75s | 5.3ms | 1000.0 |
| S2_F1000_lr1e-3 | 109.34s | 108.73s | 109.43s | 108.83s | 43.5ms | 109.44s | 4.6ms | 1000.0 |
| S2_F1500_lr3e-4 | 258.98s | 245.00s | 259.24s | 245.26s | 83.7ms | 259.25s | 6.5ms | 1500.0 |
| S2_F1500_lr3e-4 | 165.87s | 165.09s | 165.98s | 165.20s | 44.2ms | 165.99s | 5.6ms | 1500.0 |
| S2_F1500_lr3e-4 | 169.13s | 164.98s | 169.26s | 165.11s | 65.0ms | 169.27s | 6.4ms | 1500.0 |
| S2_F1500_lr3e-4 | 169.67s | 164.66s | 169.81s | 164.79s | 78.9ms | 169.82s | 6.0ms | 1500.0 |
| S2_F1500_lr3e-4 | 166.75s | 165.38s | 166.88s | 165.51s | 66.5ms | 166.90s | 5.8ms | 1500.0 |
| S2_F1500_lr3e-4 | 167.47s | 165.19s | 167.60s | 165.32s | 69.5ms | 167.61s | 5.9ms | 1500.0 |
| S2_F1500_lr3e-4 | 166.92s | 164.79s | 167.06s | 164.93s | 68.9ms | 167.08s | 5.8ms | 1500.0 |
| S2_F1500_lr3e-4 | 166.53s | 164.93s | 166.65s | 165.06s | 45.0ms | 166.66s | 5.0ms | 1500.0 |
| S2_F1500_lr3e-4 | 166.32s | 161.94s | 166.40s | 162.02s | 34.5ms | 166.41s | 4.6ms | 1500.0 |
| S2_F1500_lr3e-4 | 163.09s | 162.45s | 163.18s | 162.54s | 36.3ms | 163.18s | 4.7ms | 1500.0 |
| S2_baseline_lr1e-4_e2500 | 366.21s | 350.16s | 366.40s | 350.34s | 46.6ms | 366.41s | 5.3ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.63s | 272.90s | 273.73s | 272.99s | 41.0ms | 273.74s | 5.3ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.62s | 273.01s | 273.76s | 273.15s | 64.8ms | 273.77s | 6.2ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.73s | 273.06s | 273.86s | 273.18s | 50.8ms | 273.87s | 5.9ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 274.05s | 273.41s | 274.16s | 273.52s | 41.5ms | 274.17s | 7.5ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 274.50s | 273.91s | 274.61s | 274.02s | 46.1ms | 274.62s | 5.7ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.07s | 272.54s | 273.23s | 272.70s | 68.4ms | 273.24s | 6.0ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.92s | 273.10s | 274.04s | 273.22s | 43.8ms | 274.05s | 5.8ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.64s | 273.02s | 273.75s | 273.13s | 40.9ms | 273.76s | 5.1ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 273.86s | 273.16s | 273.96s | 273.27s | 43.2ms | 273.97s | 5.6ms | 2500.0 |

## Grouped Totals

| candidate | optimizer loop | non-loop training | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | DVF file write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_A_lr1e-2 | 78.12s | 1.66s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 90.2ms |
| D2_A_lr1e-2 | 12.65s | 18.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 71.1ms |
| D2_A_lr1e-2 | 11.55s | 11.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 69.3ms |
| D2_A_lr1e-2 | 17.50s | 14.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 72.1ms |
| D2_A_lr1e-2 | 12.38s | 11.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 48.0ms |
| D2_A_lr1e-2 | 13.54s | 8.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 96.6ms |
| D2_A_lr1e-2 | 12.83s | 8.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 68.8ms |
| D2_A_lr1e-2 | 14.50s | 21.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 69.6ms |
| D2_A_lr1e-2 | 11.40s | 14.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 96.1ms |
| D2_A_lr1e-2 | 15.64s | 14.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 98.0ms |
| D2_A_lr3e-2 | 73.52s | 1.55s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 90.6ms |
| D2_A_lr3e-2 | 6.99s | 9.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 65.1ms |
| D2_A_lr3e-2 | 7.17s | 11.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 86.1ms |
| D2_A_lr3e-2 | 7.69s | 15.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 90.9ms |
| D2_A_lr3e-2 | 8.43s | 13.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 92.9ms |
| D2_A_lr3e-2 | 7.38s | 11.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 88.5ms |
| D2_A_lr3e-2 | 8.33s | 11.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 72.6ms |
| D2_A_lr3e-2 | 8.59s | 10.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 65.8ms |
| D2_A_lr3e-2 | 7.43s | 12.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 78.3ms |
| D2_A_lr3e-2 | 8.10s | 11.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 76.9ms |
| D2_F400_lr3e-2 | 81.50s | 1.76s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 99.3ms |
| D2_F400_lr3e-2 | 13.68s | 11.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 53.9ms |
| D2_F400_lr3e-2 | 13.53s | 11.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 56.5ms |
| D2_F400_lr3e-2 | 13.69s | 17.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 88.6ms |
| D2_F400_lr3e-2 | 14.09s | 15.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 81.9ms |
| D2_F400_lr3e-2 | 13.95s | 11.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 89.0ms |
| D2_F400_lr3e-2 | 14.79s | 16.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.9ms | 75.9ms |
| D2_F400_lr3e-2 | 14.43s | 19.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 2.0ms | 79.4ms |
| D2_F400_lr3e-2 | 13.95s | 33.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 77.6ms |
| D2_F400_lr3e-2 | 14.37s | 27.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 72.5ms |
| D2_F600_lr1e-2 | 85.66s | 1.52s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 44.7ms |
| D2_F600_lr1e-2 | 20.33s | 10.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 45.6ms |
| D2_F600_lr1e-2 | 20.32s | 10.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 46.0ms |
| D2_F600_lr1e-2 | 20.83s | 14.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 82.3ms |
| D2_F600_lr1e-2 | 20.85s | 14.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 76.7ms |
| D2_F600_lr1e-2 | 20.75s | 18.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 80.3ms |
| D2_F600_lr1e-2 | 20.28s | 14.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 81.0ms |
| D2_F600_lr1e-2 | 20.39s | 12.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 56.1ms |
| D2_F600_lr1e-2 | 20.81s | 10.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 56.6ms |
| D2_F600_lr1e-2 | 20.34s | 19.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 56.0ms |
| D2_baseline_lr1e-3_e2500 | 154.64s | 1.60s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 79.7ms |
| D2_baseline_lr1e-3_e2500 | 85.70s | 40.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 54.0ms |
| D2_baseline_lr1e-3_e2500 | 83.77s | 43.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 78.7ms |
| D2_baseline_lr1e-3_e2500 | 85.21s | 52.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 54.6ms |
| D2_baseline_lr1e-3_e2500 | 84.93s | 45.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 53.3ms |
| D2_baseline_lr1e-3_e2500 | 84.47s | 30.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 51.7ms |
| D2_baseline_lr1e-3_e2500 | 85.16s | 32.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 72.8ms |
| D2_baseline_lr1e-3_e2500 | 85.33s | 51.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 2.1ms | 80.5ms |
| D2_baseline_lr1e-3_e2500 | 85.36s | 45.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 70.3ms |
| D2_baseline_lr1e-3_e2500 | 85.14s | 30.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 76.4ms |
| S2_A_lr1e-3 | 181.82s | 1.48s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 41.3ms |
| S2_A_lr1e-3 | 86.16s | 45.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 43.3ms |
| S2_A_lr1e-3 | 156.62s | 49.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 46.9ms |
| S2_A_lr1e-3 | 106.29s | 45.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 42.1ms |
| S2_A_lr1e-3 | 112.22s | 45.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 42.0ms |
| S2_A_lr1e-3 | 122.74s | 45.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 39.7ms |
| S2_A_lr1e-3 | 126.17s | 46.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 42.5ms |
| S2_A_lr1e-3 | 99.16s | 43.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 67.9ms |
| S2_A_lr1e-3 | 83.75s | 39.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 42.3ms |
| S2_A_lr1e-3 | 128.12s | 46.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 41.7ms |
| S2_F1000_lr1e-3 | 160.35s | 808.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 34.9ms |
| S2_F1000_lr1e-3 | 107.64s | 38.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 34.3ms |
| S2_F1000_lr1e-3 | 109.32s | 44.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 45.3ms |
| S2_F1000_lr1e-3 | 108.57s | 49.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 85.0ms |
| S2_F1000_lr1e-3 | 108.85s | 45.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 45.6ms |
| S2_F1000_lr1e-3 | 108.06s | 45.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 44.2ms |
| S2_F1000_lr1e-3 | 108.95s | 43.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 62.3ms |
| S2_F1000_lr1e-3 | 108.29s | 44.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 43.8ms |
| S2_F1000_lr1e-3 | 108.98s | 52.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 29.4ms | 67.1ms |
| S2_F1000_lr1e-3 | 108.69s | 47.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 43.5ms |
| S2_F1500_lr3e-4 | 243.21s | 1.50s | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 83.7ms |
| S2_F1500_lr3e-4 | 165.04s | 52.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.8ms | 44.2ms |
| S2_F1500_lr3e-4 | 164.92s | 59.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 65.0ms |
| S2_F1500_lr3e-4 | 164.61s | 52.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 78.9ms |
| S2_F1500_lr3e-4 | 165.31s | 60.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.9ms | 66.5ms |
| S2_F1500_lr3e-4 | 165.14s | 51.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.7ms | 69.5ms |
| S2_F1500_lr3e-4 | 164.72s | 62.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 68.9ms |
| S2_F1500_lr3e-4 | 164.87s | 61.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.7ms | 45.0ms |
| S2_F1500_lr3e-4 | 161.89s | 46.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 34.5ms |
| S2_F1500_lr3e-4 | 162.41s | 43.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 36.3ms |
| S2_baseline_lr1e-4_e2500 | 348.14s | 1.53s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 46.6ms |
| S2_baseline_lr1e-4_e2500 | 272.84s | 56.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 41.0ms |
| S2_baseline_lr1e-4_e2500 | 272.95s | 61.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 64.8ms |
| S2_baseline_lr1e-4_e2500 | 272.98s | 74.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 50.8ms |
| S2_baseline_lr1e-4_e2500 | 273.34s | 63.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 41.5ms |
| S2_baseline_lr1e-4_e2500 | 273.85s | 61.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 46.1ms |
| S2_baseline_lr1e-4_e2500 | 272.47s | 73.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 68.4ms |
| S2_baseline_lr1e-4_e2500 | 273.02s | 79.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 43.8ms |
| S2_baseline_lr1e-4_e2500 | 272.94s | 71.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 40.9ms |
| S2_baseline_lr1e-4_e2500 | 273.10s | 60.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 43.2ms |

## Worked Numeric Examples

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 29.58
  + initial_dense_grid/small setup 0.46
  + stage 1 pre 1.09
  + stage 1 optimizer 78.12
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 109.26

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.26
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.63
  = 109.98
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 3.07
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 12.65
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.74

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.74
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 15.86
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.82
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 11.55
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 12.39

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 12.39
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 12.51
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.64
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 17.50
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 18.16

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 18.16
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 18.28
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.59
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 12.38
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 12.98

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 12.98
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 13.07
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.71
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.54
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 14.27

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.27
  + dvf_file_write 0.10
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 14.42
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.68
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 12.83
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 13.52

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 13.52
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 13.65
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 14.50
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.17

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.17
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 15.32
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.84
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 11.40
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 12.26

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 12.26
  + dvf_file_write 0.10
  + small file-prep/post-DVF overhead before file-ready 0.09
  = 12.44
```

```text
D2_A_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 15.64
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 16.20

D2_A_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 16.20
  + dvf_file_write 0.10
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 16.38
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 13.03
  + initial_dense_grid/small setup 1.73
  + stage 1 pre 1.25
  + stage 1 optimizer 73.52
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 89.53

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 89.53
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.37
  = 89.99
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.60
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.99
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 7.61

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 7.61
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 7.74
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.83
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 7.17
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 8.01

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 8.01
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 8.18
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 1.25
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 7.69
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 8.95

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 8.95
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 9.12
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.84
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 8.43
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 9.28

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 9.28
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 9.46
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.80
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 7.38
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 8.19

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 8.19
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 8.35
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.86
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 8.33
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 9.20

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 9.20
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 9.33
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.77
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 8.59
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.00
  = 9.37

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 9.37
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 9.49
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.99
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 7.43
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 8.43

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 8.43
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 8.58
```

```text
D2_A_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.72
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 8.10
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 8.84

D2_A_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 8.84
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 8.99
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 17.02
  + initial_dense_grid/small setup 0.24
  + stage 1 pre 1.49
  + stage 1 optimizer 81.50
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 100.25

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 100.25
  + dvf_file_write 0.10
  + small file-prep/post-DVF overhead before file-ready 0.33
  = 100.68
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.72
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.68
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.42

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.42
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 14.53
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.63
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.53
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.17

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.17
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 14.29
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.69
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.35

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.35
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 14.52
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.73
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 14.09
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.84

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.84
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 15.00
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.80
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 13.95
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.77

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.77
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 14.92
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.76
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 14.79
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.56

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.56
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 15.70
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 0.56
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 14.43
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.01

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.01
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 15.15
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 1.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.02
  + stage 1 optimizer 13.95
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.52

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.52
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 15.66
```

```text
D2_F400_lr3e-2 full_script_to_dvf_in_memory
  = setup 4.26
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.02
  + stage 1 optimizer 14.37
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 18.66

D2_F400_lr3e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 18.66
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 18.80
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 16.37
  + initial_dense_grid/small setup 0.08
  + stage 1 pre 1.24
  + stage 1 optimizer 85.66
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 103.35

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 103.35
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.31
  = 103.71
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.68
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.33
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.02

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.02
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 21.10
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.32
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 20.98

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 20.98
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 21.07
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.57
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.83
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.42

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.42
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 21.55
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.62
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.85
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.49

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.49
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 21.62
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.58
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.75
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.34

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.34
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 21.49
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.69
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.28
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 20.98

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 20.98
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 21.12
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.61
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.39
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.01

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.01
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 21.13
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.72
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 20.81
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.54

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.54
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 21.66
```

```text
D2_F600_lr1e-2 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 20.34
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 21.01

D2_F600_lr1e-2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 21.01
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 21.13
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 16.16
  + initial_dense_grid/small setup 0.46
  + stage 1 pre 1.27
  + stage 1 optimizer 154.64
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 172.57

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 172.57
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.37
  = 173.02
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.67
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 85.70
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 86.41

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 86.41
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 86.53
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.62
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 83.77
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 84.43

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 84.43
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 84.59
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.69
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 85.21
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 85.95

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 85.95
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 86.08
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 1.20
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 84.93
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 86.17

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 86.17
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 86.29
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.72
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 84.47
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 85.22

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 85.22
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 85.33
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.67
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 85.16
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 85.87

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 85.87
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 86.01
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 85.33
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 85.93

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 85.93
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 86.09
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.79
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 85.36
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 86.19

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 86.19
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 86.35
```

```text
D2_baseline_lr1e-3_e2500 full_script_to_dvf_in_memory
  = setup 0.50
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 85.14
  + stage 1 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 85.67

D2_baseline_lr1e-3_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 85.67
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 85.81
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 35.62
  + initial_dense_grid/small setup 2.06
  + stage 1 pre 1.36
  + stage 1 optimizer 181.82
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 220.90

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 220.90
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.12
  = 221.06
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 86.16
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 86.75

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 86.75
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 86.85
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.74
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 156.62
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 157.41

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 157.41
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 157.51
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.29
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 106.78

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 106.78
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 106.86
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 112.22
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 112.81

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 112.81
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 112.90
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.64
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 122.74
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 123.43

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 123.43
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 123.51
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 126.17
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 126.86

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 126.86
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 126.95
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 99.16
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 99.86

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 99.86
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 99.97
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 83.75
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 84.23

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 84.23
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 84.31
```

```text
S2_A_lr1e-3 full_script_to_dvf_in_memory
  = setup 1.05
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 128.12
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 129.22

S2_A_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 129.22
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 129.30
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 25.17
  + initial_dense_grid/small setup 0.06
  + stage 1 pre 0.71
  + stage 1 optimizer 160.35
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 186.33

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 186.33
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.12
  = 186.49
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 107.64
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 108.08

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 108.08
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 108.16
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 109.32
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 109.82

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.82
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 109.92
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.86
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.57
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 109.48

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.48
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 109.63
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.78
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.85
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 109.67

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.67
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 109.77
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.53
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.06
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 108.63

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 108.63
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 108.73
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.65
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.95
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 109.64

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.64
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 109.78
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.77
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.29
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 109.11

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.11
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 109.21
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.58
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.98
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 109.61

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.61
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 109.74
```

```text
S2_F1000_lr1e-3 full_script_to_dvf_in_memory
  = setup 0.60
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 108.69
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 109.34

S2_F1000_lr1e-3 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 109.34
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 109.43
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 13.98
  + initial_dense_grid/small setup 0.39
  + stage 1 pre 1.34
  + stage 1 optimizer 243.21
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 258.98

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 258.98
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.18
  = 259.24
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 0.78
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 165.04
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 165.87

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 165.87
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 165.98
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 4.15
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 164.92
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 169.13

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 169.13
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 169.26
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 5.01
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 164.61
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 169.67

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 169.67
  + dvf_file_write 0.08
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 169.81
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 1.37
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 165.31
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 166.75

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 166.75
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 166.88
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 2.28
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 165.14
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 167.47

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 167.47
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 167.60
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 2.13
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 164.72
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 166.92

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 166.92
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 167.06
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 1.60
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 164.87
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 166.53

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 166.53
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 166.65
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 4.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 161.89
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 166.32

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 166.32
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 166.40
```

```text
S2_F1500_lr3e-4 full_script_to_dvf_in_memory
  = setup 0.64
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 162.41
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.01
  = 163.09

S2_F1500_lr3e-4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 163.09
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 163.18
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 16.06
  + initial_dense_grid/small setup 0.57
  + stage 1 pre 1.39
  + stage 1 optimizer 348.14
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 366.21

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 366.21
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.14
  = 366.40
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.74
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 272.84
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 273.63

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.63
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 273.73
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.61
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 272.95
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 273.62

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.62
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 273.76
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.68
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 272.98
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.04
  = 273.73

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.73
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 273.86
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.64
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 273.34
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 274.05

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 274.05
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 274.16
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.59
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 273.85
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 274.50

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 274.50
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 274.61
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.53
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 272.47
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.04
  = 273.07

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.07
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 273.23
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.82
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.01
  + stage 1 optimizer 273.02
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.04
  = 273.92

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.92
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 274.04
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.62
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 272.94
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.04
  = 273.64

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.64
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 273.75
```

```text
S2_baseline_lr1e-4_e2500 full_script_to_dvf_in_memory
  = setup 0.69
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 273.10
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 273.86

S2_baseline_lr1e-4_e2500 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 273.86
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 273.96
```

## Previous Conservative Timing Comparison

Positive values mean the new production path is faster than the previous conservative DVF-only timing.
Endpoint rows are the primary comparison. Component rows can shift because skipping earlier dense expansions also changes where CUDA warmup/autotune cost appears.

| candidate | metric | previous | production | previous - production |
|---|---|---:|---:|---:|
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 172.57s | -87.88s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 156.42s | -72.37s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 173.02s | -88.18s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 156.86s | -72.67s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 1.60s | -1.54s |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 154.64s | -70.65s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 173.04s | -88.19s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 86.41s | -1.72s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.74s | -1.70s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.53s | -1.68s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.86s | -1.66s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 40.1ms | 12.6ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.70s | -1.71s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.55s | -1.69s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 84.43s | 260.6ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 83.81s | 231.4ms |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 84.59s | 248.7ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 83.97s | 219.5ms |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 43.9ms | 8.8ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 83.77s | 222.5ms |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 84.63s | 228.9ms |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 85.95s | -1.26s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.27s | -1.22s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.08s | -1.23s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.39s | -1.20s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 52.2ms | 0.5ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.21s | -1.22s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.23s | -1.37s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 86.17s | -1.47s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.48s | -1.44s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.29s | -1.45s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.60s | -1.41s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 45.5ms | 7.2ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 84.93s | -935.7ms |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.33s | -1.47s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 85.22s | -528.8ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 84.51s | -461.1ms |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 85.33s | -490.9ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 84.62s | -423.2ms |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 30.4ms | 22.4ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 84.47s | -483.1ms |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 85.86s | -1.01s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 85.87s | -1.17s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.19s | -1.15s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.01s | -1.17s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.34s | -1.15s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 32.9ms | 19.8ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.16s | -1.17s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.04s | -1.19s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 85.93s | -1.24s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.38s | -1.34s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.09s | -1.25s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.55s | -1.35s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 51.0ms | 1.7ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.33s | -1.34s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.12s | -1.26s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 86.19s | -1.50s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.40s | -1.36s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 86.35s | -1.50s |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.56s | -1.36s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 45.8ms | 6.9ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.36s | -1.37s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 86.38s | -1.52s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_in_memory | 84.70s | 85.67s | -976.8ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_in_memory | 84.04s | 85.17s | -1.12s |
| D2_baseline_lr1e-3_e2500 | full_script_to_dvf_file | 84.84s | 85.81s | -967.0ms |
| D2_baseline_lr1e-3_e2500 | loaded_images_to_dvf_file | 84.19s | 85.31s | -1.11s |
| D2_baseline_lr1e-3_e2500 | total_non_loop_training | 52.7ms | 30.5ms | 22.2ms |
| D2_baseline_lr1e-3_e2500 | total_optimizer_loop | 83.99s | 85.14s | -1.15s |
| D2_baseline_lr1e-3_e2500 | external_wall | 84.86s | 85.84s | -984.0ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 366.21s | -92.36s |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 350.16s | -76.99s |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 366.40s | -92.44s |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 350.34s | -77.08s |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 1.53s | -1.47s |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 348.14s | -75.04s |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 366.41s | -92.44s |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.63s | 221.9ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 272.90s | 263.6ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.73s | 230.8ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 272.99s | 272.4ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 56.8ms | 4.0ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 272.84s | 259.6ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.74s | 231.1ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.62s | 239.1ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.01s | 152.0ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.76s | 200.9ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.15s | 113.8ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 61.3ms | -0.5ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 272.95s | 152.5ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.77s | 198.9ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.73s | 125.5ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.06s | 107.6ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.86s | 100.0ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.18s | 82.1ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 74.0ms | -13.1ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 272.98s | 120.5ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.87s | 99.1ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 274.05s | -196.7ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.41s | -247.0ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 274.16s | -200.0ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.52s | -250.3ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 63.5ms | -2.6ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 273.34s | -243.8ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 274.17s | -202.5ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 274.50s | -644.3ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.91s | -747.8ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 274.61s | -650.0ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 274.02s | -753.5ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 61.5ms | -0.7ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 273.85s | -746.5ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 274.62s | -650.4ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.07s | 781.8ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 272.54s | 619.2ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.23s | 733.6ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 272.70s | 571.0ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 73.9ms | -13.1ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 272.47s | 632.0ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.24s | 732.8ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.92s | -59.1ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.10s | 66.4ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 274.04s | -82.0ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.22s | 43.4ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 79.2ms | -18.4ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 273.02s | 85.0ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 274.05s | -83.3ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.64s | 217.4ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.02s | 147.0ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.75s | 208.8ms |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.13s | 138.4ms |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 71.5ms | -10.6ms |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 272.94s | 157.1ms |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.76s | 208.2ms |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_in_memory | 273.86s | 273.86s | 0.00s |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_in_memory | 273.16s | 273.16s | 0.00s |
| S2_baseline_lr1e-4_e2500 | full_script_to_dvf_file | 273.96s | 273.96s | 0.00s |
| S2_baseline_lr1e-4_e2500 | loaded_images_to_dvf_file | 273.27s | 273.27s | 0.00s |
| S2_baseline_lr1e-4_e2500 | total_non_loop_training | 60.9ms | 60.9ms | 0.00s |
| S2_baseline_lr1e-4_e2500 | total_optimizer_loop | 273.10s | 273.10s | 0.00s |
| S2_baseline_lr1e-4_e2500 | external_wall | 273.97s | 273.97s | 0.00s |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 103.35s | -82.34s |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 86.98s | -66.62s |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 103.71s | -82.57s |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 87.34s | -66.86s |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 1.52s | -1.50s |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 85.66s | -65.32s |
| D2_F600_lr1e-2 | external_wall | 21.14s | 103.72s | -82.57s |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.02s | -0.7ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.34s | 24.0ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.10s | 33.6ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.42s | 58.3ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 10.5ms | 9.2ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.33s | 15.0ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.11s | 35.1ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 20.98s | 32.4ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.33s | 33.4ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.07s | 59.9ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.42s | 60.9ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 10.8ms | 9.0ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.32s | 24.4ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.08s | 61.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.42s | -401.1ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.84s | -481.0ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.55s | -416.0ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.98s | -496.0ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 14.1ms | 5.7ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.83s | -487.0ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.56s | -416.1ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.49s | -472.1ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.86s | -502.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.62s | -489.4ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 21.00s | -520.1ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 14.8ms | 5.0ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.85s | -508.2ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.63s | -491.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.34s | -326.0ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.77s | -404.6ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.49s | -357.7ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.92s | -436.3ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 18.0ms | 1.7ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.75s | -407.2ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.50s | -359.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 20.98s | 34.0ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.29s | 69.0ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.12s | 8.5ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.44s | 43.6ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 14.8ms | 4.9ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.28s | 64.1ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.14s | 3.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.01s | 0.8ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.40s | -38.3ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.13s | 5.9ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.51s | -33.2ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 12.0ms | 7.7ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.39s | -46.2ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.14s | 6.1ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.54s | -528.6ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.82s | -463.3ms |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.66s | -526.5ms |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.94s | -461.2ms |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 10.9ms | 8.8ms |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.81s | -472.4ms |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.67s | -525.9ms |
| D2_F600_lr1e-2 | full_script_to_dvf_in_memory | 21.01s | 21.01s | 0.00s |
| D2_F600_lr1e-2 | loaded_images_to_dvf_in_memory | 20.36s | 20.36s | 0.00s |
| D2_F600_lr1e-2 | full_script_to_dvf_file | 21.13s | 21.13s | 0.00s |
| D2_F600_lr1e-2 | loaded_images_to_dvf_file | 20.48s | 20.48s | 0.00s |
| D2_F600_lr1e-2 | total_non_loop_training | 19.7ms | 19.7ms | 0.00s |
| D2_F600_lr1e-2 | total_optimizer_loop | 20.34s | 20.34s | 0.00s |
| D2_F600_lr1e-2 | external_wall | 21.14s | 21.14s | 0.00s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 100.25s | -81.59s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 83.23s | -68.84s |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 100.68s | -81.88s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 83.67s | -69.12s |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 1.76s | -1.74s |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 81.50s | -67.13s |
| D2_F400_lr3e-2 | external_wall | 18.82s | 100.69s | -81.88s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 14.42s | 4.24s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 13.69s | 703.3ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 14.53s | 4.28s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 13.80s | 739.6ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 11.9ms | 15.5ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 13.68s | 687.9ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 14.53s | 4.28s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 14.17s | 4.49s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 13.54s | 854.7ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 14.29s | 4.52s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 13.66s | 887.1ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 11.3ms | 16.1ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 13.53s | 838.6ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 14.29s | 4.52s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 14.35s | 4.30s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 13.71s | 687.1ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 14.52s | 4.28s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 13.88s | 667.1ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 17.5ms | 9.9ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 13.69s | 677.2ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 14.53s | 4.28s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 14.84s | 3.82s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 14.11s | 286.6ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 15.00s | 3.81s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.26s | 278.8ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 15.8ms | 11.6ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 14.09s | 274.6ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 15.01s | 3.81s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 14.77s | 3.89s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 13.97s | 429.8ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 14.92s | 3.88s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.12s | 423.2ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 11.4ms | 16.0ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 13.95s | 414.4ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 14.94s | 3.88s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 15.56s | 3.09s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 14.80s | -408.0ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 15.70s | 3.10s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.94s | -402.4ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 16.5ms | 10.8ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 14.79s | -417.4ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 15.72s | 3.10s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 15.01s | 3.65s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 14.45s | -50.1ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 15.15s | 3.66s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.59s | -44.5ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 19.9ms | 7.4ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 14.43s | -58.3ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 15.16s | 3.66s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 15.52s | 3.13s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 13.98s | 414.3ms |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 15.66s | 3.14s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.12s | 420.0ms |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 33.1ms | -5.8ms |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 13.95s | 420.5ms |
| D2_F400_lr3e-2 | external_wall | 18.82s | 15.68s | 3.14s |
| D2_F400_lr3e-2 | full_script_to_dvf_in_memory | 18.66s | 18.66s | 0.00s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_in_memory | 14.40s | 14.40s | 0.00s |
| D2_F400_lr3e-2 | full_script_to_dvf_file | 18.80s | 18.80s | 0.00s |
| D2_F400_lr3e-2 | loaded_images_to_dvf_file | 14.54s | 14.54s | 0.00s |
| D2_F400_lr3e-2 | total_non_loop_training | 27.3ms | 27.3ms | 0.00s |
| D2_F400_lr3e-2 | total_optimizer_loop | 14.37s | 14.37s | 0.00s |
| D2_F400_lr3e-2 | external_wall | 18.82s | 18.82s | 0.00s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 109.26s | -93.06s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 79.68s | -64.03s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 109.98s | -93.60s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 80.39s | -64.56s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 1.66s | -1.64s |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 78.12s | -62.49s |
| D2_A_lr1e-2 | external_wall | 16.39s | 109.99s | -93.60s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 15.74s | 461.3ms |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 12.67s | 2.98s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 15.86s | 516.7ms |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 12.79s | 3.04s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 18.0ms | -3.1ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 12.65s | 2.99s |
| D2_A_lr1e-2 | external_wall | 16.39s | 15.88s | 515.9ms |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 12.39s | 3.81s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 11.56s | 4.09s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 12.51s | 3.87s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 11.68s | 4.14s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 11.2ms | 3.7ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 11.55s | 4.08s |
| D2_A_lr1e-2 | external_wall | 16.39s | 12.52s | 3.88s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 18.16s | -1.96s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 17.51s | -1.86s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 18.28s | -1.91s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 17.64s | -1.81s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 14.8ms | 0.1ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 17.50s | -1.86s |
| D2_A_lr1e-2 | external_wall | 16.39s | 18.30s | -1.90s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 12.98s | 3.22s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 12.39s | 3.26s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 13.07s | 3.31s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 12.48s | 3.35s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 11.7ms | 3.2ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 12.38s | 3.26s |
| D2_A_lr1e-2 | external_wall | 16.39s | 13.07s | 3.32s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 14.27s | 1.93s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 13.55s | 2.10s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 14.42s | 1.96s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 13.70s | 2.12s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 8.7ms | 6.2ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 13.54s | 2.09s |
| D2_A_lr1e-2 | external_wall | 16.39s | 14.43s | 1.97s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 13.52s | 2.68s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 12.84s | 2.81s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 13.65s | 2.73s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 12.97s | 2.86s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 8.3ms | 6.6ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 12.83s | 2.81s |
| D2_A_lr1e-2 | external_wall | 16.39s | 13.66s | 2.73s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 15.17s | 1.03s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 14.52s | 1.13s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 15.32s | 1.05s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 14.67s | 1.16s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 21.4ms | -6.5ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 14.50s | 1.14s |
| D2_A_lr1e-2 | external_wall | 16.39s | 15.33s | 1.06s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 12.26s | 3.94s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 11.42s | 4.23s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 12.44s | 3.94s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 11.60s | 4.23s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 14.7ms | 0.2ms |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 11.40s | 4.23s |
| D2_A_lr1e-2 | external_wall | 16.39s | 12.46s | 3.93s |
| D2_A_lr1e-2 | full_script_to_dvf_in_memory | 16.20s | 16.20s | 0.00s |
| D2_A_lr1e-2 | loaded_images_to_dvf_in_memory | 15.65s | 15.65s | 0.00s |
| D2_A_lr1e-2 | full_script_to_dvf_file | 16.38s | 16.38s | 0.00s |
| D2_A_lr1e-2 | loaded_images_to_dvf_file | 15.83s | 15.83s | 0.00s |
| D2_A_lr1e-2 | total_non_loop_training | 14.9ms | 14.9ms | 0.00s |
| D2_A_lr1e-2 | total_optimizer_loop | 15.64s | 15.64s | 0.00s |
| D2_A_lr1e-2 | external_wall | 16.39s | 16.39s | 0.00s |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 89.53s | -80.69s |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 76.50s | -68.38s |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 89.99s | -81.00s |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 76.96s | -68.69s |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 1.55s | -1.54s |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 73.52s | -65.42s |
| D2_A_lr3e-2 | external_wall | 9.00s | 90.00s | -81.00s |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 7.61s | 1.23s |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 7.00s | 1.11s |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 7.74s | 1.25s |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 7.14s | 1.13s |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 9.3ms | 2.5ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 6.99s | 1.11s |
| D2_A_lr3e-2 | external_wall | 9.00s | 7.75s | 1.25s |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 8.01s | 824.8ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 7.18s | 936.6ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 8.18s | 816.8ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 7.34s | 928.6ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 11.0ms | 0.8ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 7.17s | 935.8ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 8.19s | 818.0ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 8.95s | -117.4ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 7.71s | 409.4ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 9.12s | -130.4ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 7.87s | 396.4ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 15.9ms | -4.0ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 7.69s | 409.7ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 9.14s | -131.7ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 9.28s | -446.1ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 8.44s | -328.5ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 9.46s | -465.5ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 8.62s | -348.0ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 13.5ms | -1.6ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 8.43s | -326.8ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 9.47s | -467.9ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 8.19s | 642.8ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 7.40s | 720.0ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 8.35s | 637.6ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 7.56s | 714.8ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 11.8ms | -0.0ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 7.38s | 720.0ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 8.37s | 636.5ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 9.20s | -365.8ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 8.34s | -227.3ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 9.33s | -337.4ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 8.47s | -198.9ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 11.5ms | 0.3ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 8.33s | -227.0ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 9.34s | -336.4ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 9.37s | -535.2ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 8.60s | -488.0ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 9.49s | -500.8ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 8.72s | -453.6ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 10.4ms | 1.4ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 8.59s | -489.2ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 9.50s | -499.0ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 8.43s | 409.3ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 7.44s | 672.4ms |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 8.58s | 413.5ms |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 7.59s | 676.7ms |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 12.1ms | -0.3ms |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 7.43s | 675.2ms |
| D2_A_lr3e-2 | external_wall | 9.00s | 8.59s | 413.6ms |
| D2_A_lr3e-2 | full_script_to_dvf_in_memory | 8.84s | 8.84s | 0.00s |
| D2_A_lr3e-2 | loaded_images_to_dvf_in_memory | 8.12s | 8.12s | 0.00s |
| D2_A_lr3e-2 | full_script_to_dvf_file | 8.99s | 8.99s | 0.00s |
| D2_A_lr3e-2 | loaded_images_to_dvf_file | 8.27s | 8.27s | 0.00s |
| D2_A_lr3e-2 | total_non_loop_training | 11.8ms | 11.8ms | 0.00s |
| D2_A_lr3e-2 | total_optimizer_loop | 8.10s | 8.10s | 0.00s |
| D2_A_lr3e-2 | external_wall | 9.00s | 9.00s | 0.00s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 258.98s | -95.89s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 245.00s | -82.55s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 259.24s | -96.06s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 245.26s | -82.72s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 1.50s | -1.46s |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 243.21s | -80.80s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 259.25s | -96.07s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 165.87s | -2.78s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 165.09s | -2.64s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 165.98s | -2.80s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 165.20s | -2.66s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 52.3ms | -8.6ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 165.04s | -2.63s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 165.99s | -2.80s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 169.13s | -6.03s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 164.98s | -2.53s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 169.26s | -6.08s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 165.11s | -2.58s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 59.9ms | -16.3ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 164.92s | -2.51s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 169.27s | -6.09s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 169.67s | -6.58s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 164.66s | -2.21s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 169.81s | -6.63s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 164.79s | -2.26s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 52.0ms | -8.4ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 164.61s | -2.20s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 169.82s | -6.63s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 166.75s | -3.66s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 165.38s | -2.92s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 166.88s | -3.71s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 165.51s | -2.97s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 60.7ms | -17.0ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 165.31s | -2.91s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 166.90s | -3.71s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 167.47s | -4.38s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 165.19s | -2.74s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 167.60s | -4.42s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 165.32s | -2.78s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 51.7ms | -8.1ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 165.14s | -2.73s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 167.61s | -4.43s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 166.92s | -3.83s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 164.79s | -2.33s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 167.06s | -3.89s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 164.93s | -2.40s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 62.2ms | -18.6ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 164.72s | -2.32s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 167.08s | -3.89s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 166.53s | -3.44s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 164.93s | -2.48s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 166.65s | -3.48s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 165.06s | -2.52s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 61.4ms | -17.7ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 164.87s | -2.46s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 166.66s | -3.48s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 166.32s | -3.23s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 161.94s | 513.5ms |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 166.40s | -3.23s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 162.02s | 516.7ms |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 46.7ms | -3.1ms |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 161.89s | 517.1ms |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 166.41s | -3.23s |
| S2_F1500_lr3e-4 | full_script_to_dvf_in_memory | 163.09s | 163.09s | 0.00s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_in_memory | 162.45s | 162.45s | 0.00s |
| S2_F1500_lr3e-4 | full_script_to_dvf_file | 163.18s | 163.18s | 0.00s |
| S2_F1500_lr3e-4 | loaded_images_to_dvf_file | 162.54s | 162.54s | 0.00s |
| S2_F1500_lr3e-4 | total_non_loop_training | 43.6ms | 43.6ms | 0.00s |
| S2_F1500_lr3e-4 | total_optimizer_loop | 162.41s | 162.41s | 0.00s |
| S2_F1500_lr3e-4 | external_wall | 163.18s | 163.18s | 0.00s |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 186.33s | -77.00s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 161.17s | -52.43s |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 186.49s | -77.06s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 161.33s | -52.49s |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 808.0ms | -760.9ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 160.35s | -51.67s |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 186.50s | -77.06s |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 108.08s | 1.26s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 107.67s | 1.06s |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 108.16s | 1.28s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 107.75s | 1.08s |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 38.2ms | 8.9ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 107.64s | 1.05s |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 108.17s | 1.28s |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.82s | -486.5ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 109.36s | -627.2ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.92s | -489.7ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 109.46s | -630.4ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 44.2ms | 3.0ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 109.32s | -630.4ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.93s | -490.2ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.48s | -142.9ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 108.62s | 116.1ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.63s | -192.5ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 108.77s | 66.5ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 49.8ms | -2.7ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.57s | 118.5ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.64s | -194.9ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.67s | -335.7ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 108.89s | -156.5ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.77s | -340.2ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 108.99s | -161.0ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 45.4ms | 1.8ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.85s | -158.2ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.78s | -340.3ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 108.63s | 702.2ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 108.10s | 631.1ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 108.73s | 701.1ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 108.20s | 630.0ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 45.2ms | 1.9ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.06s | 629.4ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 108.74s | 701.2ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.64s | -305.1ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 109.00s | -261.4ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.78s | -342.5ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 109.13s | -298.7ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 43.7ms | 3.5ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.95s | -265.0ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.79s | -344.3ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.11s | 224.1ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 108.34s | 397.3ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.21s | 227.0ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 108.43s | 400.2ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 44.7ms | 2.4ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.29s | 395.2ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.22s | 227.0ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.61s | -276.9ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 109.04s | -301.5ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.74s | -308.5ms |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 109.17s | -333.1ms |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 52.6ms | -5.5ms |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.98s | -296.1ms |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.75s | -311.3ms |
| S2_F1000_lr1e-3 | full_script_to_dvf_in_memory | 109.34s | 109.34s | 0.00s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_in_memory | 108.73s | 108.73s | 0.00s |
| S2_F1000_lr1e-3 | full_script_to_dvf_file | 109.43s | 109.43s | 0.00s |
| S2_F1000_lr1e-3 | loaded_images_to_dvf_file | 108.83s | 108.83s | 0.00s |
| S2_F1000_lr1e-3 | total_non_loop_training | 47.1ms | 47.1ms | 0.00s |
| S2_F1000_lr1e-3 | total_optimizer_loop | 108.69s | 108.69s | 0.00s |
| S2_F1000_lr1e-3 | external_wall | 109.44s | 109.44s | 0.00s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 220.90s | -91.68s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 185.28s | -57.11s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 221.06s | -91.76s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 185.44s | -57.18s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 1.48s | -1.43s |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 181.82s | -53.69s |
| S2_A_lr1e-3 | external_wall | 129.31s | 221.07s | -91.76s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 86.75s | 42.47s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 86.21s | 41.96s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 86.85s | 42.46s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 86.31s | 41.95s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 45.4ms | 1.1ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 86.16s | 41.96s |
| S2_A_lr1e-3 | external_wall | 129.31s | 87.68s | 41.63s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 157.41s | -28.20s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 156.67s | -28.50s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 157.51s | -28.21s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 156.77s | -28.51s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 49.4ms | -2.9ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 156.62s | -28.50s |
| S2_A_lr1e-3 | external_wall | 129.31s | 157.52s | -28.21s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 106.78s | 22.44s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 106.34s | 21.83s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 106.86s | 22.44s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 106.43s | 21.83s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 45.5ms | 1.0ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 106.29s | 21.83s |
| S2_A_lr1e-3 | external_wall | 129.31s | 106.87s | 22.44s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 112.81s | 16.40s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 112.27s | 15.90s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 112.90s | 16.41s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 112.35s | 15.90s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 45.8ms | 0.6ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 112.22s | 15.90s |
| S2_A_lr1e-3 | external_wall | 129.31s | 112.91s | 16.40s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 123.43s | 5.79s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 122.79s | 5.38s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 123.51s | 5.79s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 122.87s | 5.39s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 45.2ms | 1.3ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 122.74s | 5.38s |
| S2_A_lr1e-3 | external_wall | 129.31s | 123.52s | 5.79s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 126.86s | 2.36s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 126.21s | 1.96s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 126.95s | 2.36s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 126.30s | 1.95s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 46.4ms | 0.0ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 126.17s | 1.96s |
| S2_A_lr1e-3 | external_wall | 129.31s | 126.96s | 2.36s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 99.86s | 29.36s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 99.21s | 28.96s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 99.97s | 29.33s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 99.32s | 28.94s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 43.7ms | 2.8ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 99.16s | 28.96s |
| S2_A_lr1e-3 | external_wall | 129.31s | 99.98s | 29.33s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 84.23s | 44.99s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 83.79s | 44.38s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 84.31s | 44.99s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 83.87s | 44.38s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 39.9ms | 6.6ms |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 83.75s | 44.37s |
| S2_A_lr1e-3 | external_wall | 129.31s | 84.32s | 44.99s |
| S2_A_lr1e-3 | full_script_to_dvf_in_memory | 129.22s | 129.22s | 0.00s |
| S2_A_lr1e-3 | loaded_images_to_dvf_in_memory | 128.17s | 128.17s | 0.00s |
| S2_A_lr1e-3 | full_script_to_dvf_file | 129.30s | 129.30s | 0.00s |
| S2_A_lr1e-3 | loaded_images_to_dvf_file | 128.26s | 128.26s | 0.00s |
| S2_A_lr1e-3 | total_non_loop_training | 46.5ms | 46.5ms | 0.00s |
| S2_A_lr1e-3 | total_optimizer_loop | 128.12s | 128.12s | 0.00s |
| S2_A_lr1e-3 | external_wall | 129.31s | 129.31s | 0.00s |

## Stage Breakdown

| candidate | stage | pre | optimizer | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | post-to-DVF | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D2_A_lr1e-2 | 1 | 1.09s | 78.12s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 331.0 |
| D2_A_lr1e-2 | 1 | 8.6ms | 12.65s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 350.0 |
| D2_A_lr1e-2 | 1 | 2.0ms | 11.55s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 325.0 |
| D2_A_lr1e-2 | 1 | 1.9ms | 17.50s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 491.0 |
| D2_A_lr1e-2 | 1 | 5.2ms | 12.38s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 355.0 |
| D2_A_lr1e-2 | 1 | 1.9ms | 13.54s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 399.0 |
| D2_A_lr1e-2 | 1 | 1.8ms | 12.83s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 372.0 |
| D2_A_lr1e-2 | 1 | 11.1ms | 14.50s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 410.0 |
| D2_A_lr1e-2 | 1 | 5.9ms | 11.40s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 314.0 |
| D2_A_lr1e-2 | 1 | 4.2ms | 15.64s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 450.0 |
| D2_A_lr3e-2 | 1 | 1.25s | 73.52s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 200.0 |
| D2_A_lr3e-2 | 1 | 2.3ms | 6.99s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 202.0 |
| D2_A_lr3e-2 | 1 | 3.5ms | 7.17s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 200.0 |
| D2_A_lr3e-2 | 1 | 3.9ms | 7.69s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 207.0 |
| D2_A_lr3e-2 | 1 | 3.6ms | 8.43s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 239.0 |
| D2_A_lr3e-2 | 1 | 4.1ms | 7.38s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 206.0 |
| D2_A_lr3e-2 | 1 | 3.9ms | 8.33s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 228.0 |
| D2_A_lr3e-2 | 1 | 3.5ms | 8.59s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 242.0 |
| D2_A_lr3e-2 | 1 | 3.5ms | 7.43s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 205.0 |
| D2_A_lr3e-2 | 1 | 3.5ms | 8.10s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 214.0 |
| D2_F400_lr3e-2 | 1 | 1.49s | 81.50s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 2.6ms | 13.68s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 3.0ms | 13.53s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 4.9ms | 13.69s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 2.4ms | 14.09s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 2.7ms | 13.95s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 3.9ms | 14.79s | 0.00s | 0.00s | 0.00s | 0.00s | 1.9ms | 1.9ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 8.0ms | 14.43s | 0.00s | 0.00s | 0.00s | 0.00s | 2.0ms | 2.0ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 22.9ms | 13.95s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 400.0 |
| D2_F400_lr3e-2 | 1 | 17.5ms | 14.37s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 400.0 |
| D2_F600_lr1e-2 | 1 | 1.24s | 85.66s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 2.1ms | 20.33s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 2.2ms | 20.32s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 1.7ms | 20.83s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 1.8ms | 20.85s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 4.3ms | 20.75s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 2.1ms | 20.28s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 1.8ms | 20.39s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 1.7ms | 20.81s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 600.0 |
| D2_F600_lr1e-2 | 1 | 10.7ms | 20.34s | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 1.4ms | 600.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 1.27s | 154.64s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 11.0ms | 85.70s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 2.0ms | 83.77s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 11.2ms | 85.21s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 3.7ms | 84.93s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 2.0ms | 84.47s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 1.9ms | 85.16s | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 1.7ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 8.3ms | 85.33s | 0.00s | 0.00s | 0.00s | 0.00s | 2.1ms | 2.1ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 3.2ms | 85.36s | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 1.8ms | 2500.0 |
| D2_baseline_lr1e-3_e2500 | 1 | 1.8ms | 85.14s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 2500.0 |
| S2_A_lr1e-3 | 1 | 1.36s | 181.82s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 1018.0 |
| S2_A_lr1e-3 | 1 | 3.0ms | 86.16s | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 28.3ms | 795.0 |
| S2_A_lr1e-3 | 1 | 4.8ms | 156.62s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1438.0 |
| S2_A_lr1e-3 | 1 | 2.6ms | 106.29s | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 28.5ms | 980.0 |
| S2_A_lr1e-3 | 1 | 4.9ms | 112.22s | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 28.4ms | 1033.0 |
| S2_A_lr1e-3 | 1 | 4.6ms | 122.74s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 1133.0 |
| S2_A_lr1e-3 | 1 | 4.6ms | 126.17s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1161.0 |
| S2_A_lr1e-3 | 1 | 4.6ms | 99.16s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 910.0 |
| S2_A_lr1e-3 | 1 | 2.3ms | 83.75s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 772.0 |
| S2_A_lr1e-3 | 1 | 5.0ms | 128.12s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1177.0 |
| S2_F1000_lr1e-3 | 1 | 714.7ms | 160.35s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 2.0ms | 107.64s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 1.9ms | 109.32s | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 28.3ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.5ms | 108.57s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.5ms | 108.85s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.6ms | 108.06s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 3.0ms | 108.95s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.7ms | 108.29s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.7ms | 108.98s | 0.00s | 0.00s | 0.00s | 0.00s | 29.4ms | 29.4ms | 1000.0 |
| S2_F1000_lr1e-3 | 1 | 4.5ms | 108.69s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 1000.0 |
| S2_F1500_lr3e-4 | 1 | 1.34s | 243.21s | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 28.5ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.9ms | 165.04s | 0.00s | 0.00s | 0.00s | 0.00s | 28.8ms | 28.8ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.8ms | 164.92s | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 28.3ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 5.4ms | 164.61s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.8ms | 165.31s | 0.00s | 0.00s | 0.00s | 0.00s | 28.9ms | 28.9ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.9ms | 165.14s | 0.00s | 0.00s | 0.00s | 0.00s | 28.7ms | 28.7ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.9ms | 164.72s | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 28.5ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 4.9ms | 164.87s | 0.00s | 0.00s | 0.00s | 0.00s | 28.7ms | 28.7ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 5.2ms | 161.89s | 0.00s | 0.00s | 0.00s | 0.00s | 28.1ms | 28.1ms | 1500.0 |
| S2_F1500_lr3e-4 | 1 | 2.1ms | 162.41s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 1500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 1.39s | 348.14s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.7ms | 272.84s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.6ms | 272.95s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.7ms | 272.98s | 0.00s | 0.00s | 0.00s | 0.00s | 28.5ms | 28.5ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 5.6ms | 273.34s | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 28.4ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.7ms | 273.85s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.7ms | 272.47s | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 28.4ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 8.6ms | 273.02s | 0.00s | 0.00s | 0.00s | 0.00s | 28.4ms | 28.4ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 3.0ms | 272.94s | 0.00s | 0.00s | 0.00s | 0.00s | 28.2ms | 28.2ms | 2500.0 |
| S2_baseline_lr1e-4_e2500 | 1 | 4.8ms | 273.10s | 0.00s | 0.00s | 0.00s | 0.00s | 28.3ms | 28.3ms | 2500.0 |

## Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_first10.csv`
