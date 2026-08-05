# OASIS DVF-Only Runtime Timing

- Phase: `first10`
- Output root: `<external-run-root>/runs/oasis_directcp842_persistent_dvf_timing_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-04_oasis_directcp842_persistent_process_dvf_timing`
- Completed command rows: 40
- Timing JSON rows: 40/40
- Dense DVF files present: 40/40
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
| A_fast_decay | 144.55s | 116.93s | 144.63s | 117.01s | 31.6ms | 144.64s | 4.4ms | 600.0 |
| A_fast_decay | 16.02s | 15.52s | 16.09s | 15.59s | 31.1ms | 16.10s | 3.1ms | 575.0 |
| A_fast_decay | 13.56s | 13.19s | 13.66s | 13.29s | 33.2ms | 13.67s | 7.6ms | 550.0 |
| A_fast_decay | 14.99s | 14.59s | 15.08s | 14.68s | 30.2ms | 15.09s | 4.5ms | 600.0 |
| A_fast_decay | 15.26s | 14.62s | 15.38s | 14.74s | 63.2ms | 15.39s | 6.0ms | 575.0 |
| A_fast_decay | 15.44s | 14.90s | 15.56s | 15.02s | 65.5ms | 15.57s | 6.1ms | 600.0 |
| A_fast_decay | 14.76s | 14.17s | 14.84s | 14.25s | 42.4ms | 14.85s | 4.9ms | 550.0 |
| A_fast_decay | 15.64s | 15.04s | 15.78s | 15.18s | 87.6ms | 15.79s | 6.0ms | 600.0 |
| A_fast_decay | 14.97s | 14.30s | 15.10s | 14.43s | 63.5ms | 15.11s | 5.9ms | 575.0 |
| A_fast_decay | 14.99s | 14.33s | 15.13s | 14.47s | 64.3ms | 15.14s | 5.9ms | 550.0 |
| original_mrdbscp | 177.41s | 147.71s | 177.48s | 147.77s | 30.9ms | 177.49s | 5.7ms | 2500.0 |
| original_mrdbscp | 62.84s | 62.51s | 62.90s | 62.58s | 29.3ms | 62.91s | 6.0ms | 2500.0 |
| original_mrdbscp | 62.29s | 61.86s | 62.36s | 61.92s | 29.6ms | 62.37s | 5.8ms | 2500.0 |
| original_mrdbscp | 62.48s | 62.08s | 62.55s | 62.15s | 30.6ms | 62.56s | 6.0ms | 2500.0 |
| original_mrdbscp | 62.53s | 62.09s | 62.61s | 62.18s | 48.7ms | 62.62s | 4.1ms | 2500.0 |
| original_mrdbscp | 62.30s | 61.93s | 62.36s | 62.00s | 31.8ms | 62.38s | 6.3ms | 2500.0 |
| original_mrdbscp | 62.36s | 61.95s | 62.50s | 62.09s | 29.1ms | 62.51s | 5.8ms | 2500.0 |
| original_mrdbscp | 62.58s | 62.11s | 62.65s | 62.17s | 29.5ms | 62.66s | 8.6ms | 2500.0 |
| original_mrdbscp | 62.51s | 62.11s | 62.60s | 62.20s | 51.2ms | 62.61s | 6.2ms | 2500.0 |
| original_mrdbscp | 62.39s | 62.00s | 62.45s | 62.06s | 29.9ms | 62.46s | 6.0ms | 2500.0 |
| sinr_cps2 | 356.39s | 327.62s | 356.53s | 327.75s | 31.5ms | 356.54s | 4.5ms | 2500.0 |
| sinr_cps2 | 271.29s | 270.86s | 271.36s | 270.93s | 28.8ms | 271.36s | 4.6ms | 2500.0 |
| sinr_cps2 | 271.65s | 271.18s | 271.72s | 271.25s | 29.8ms | 271.73s | 7.3ms | 2500.0 |
| sinr_cps2 | 271.51s | 271.12s | 271.58s | 271.20s | 29.4ms | 271.59s | 4.4ms | 2500.0 |
| sinr_cps2 | 271.37s | 270.94s | 271.44s | 271.01s | 28.7ms | 271.45s | 4.7ms | 2500.0 |
| sinr_cps2 | 271.17s | 270.62s | 271.25s | 270.70s | 31.4ms | 271.26s | 5.3ms | 2500.0 |
| sinr_cps2 | 271.33s | 270.90s | 271.40s | 270.97s | 29.2ms | 271.41s | 4.5ms | 2500.0 |
| sinr_cps2 | 271.48s | 270.93s | 271.55s | 271.00s | 29.2ms | 271.56s | 4.7ms | 2500.0 |
| sinr_cps2 | 271.34s | 270.89s | 271.41s | 270.96s | 29.5ms | 271.41s | 4.9ms | 2500.0 |
| sinr_cps2 | 271.24s | 270.82s | 271.31s | 270.89s | 28.9ms | 271.32s | 5.1ms | 2500.0 |
| sinr_cps4 | 182.49s | 155.60s | 182.81s | 155.92s | 29.6ms | 182.81s | 4.7ms | 2500.0 |
| sinr_cps4 | 107.28s | 106.82s | 107.37s | 106.92s | 30.4ms | 107.38s | 3.6ms | 2500.0 |
| sinr_cps4 | 107.49s | 107.07s | 107.60s | 107.18s | 31.0ms | 107.61s | 4.5ms | 2500.0 |
| sinr_cps4 | 107.22s | 106.82s | 107.33s | 106.92s | 30.8ms | 107.33s | 3.2ms | 2500.0 |
| sinr_cps4 | 107.42s | 106.99s | 107.51s | 107.08s | 30.9ms | 107.52s | 4.9ms | 2500.0 |
| sinr_cps4 | 107.37s | 106.96s | 107.47s | 107.06s | 31.2ms | 107.47s | 4.9ms | 2500.0 |
| sinr_cps4 | 107.44s | 107.03s | 107.53s | 107.11s | 30.4ms | 107.53s | 4.8ms | 2500.0 |
| sinr_cps4 | 107.40s | 106.95s | 107.49s | 107.05s | 28.9ms | 107.50s | 4.8ms | 2500.0 |
| sinr_cps4 | 107.47s | 107.06s | 107.59s | 107.17s | 44.1ms | 107.59s | 4.8ms | 2500.0 |
| sinr_cps4 | 107.38s | 107.03s | 107.47s | 107.12s | 28.9ms | 107.48s | 4.6ms | 2500.0 |

## Grouped Totals

| candidate | optimizer loop | non-loop training | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | DVF file write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 104.27s | 12.27s | 1.5ms | 0.00s | 0.00s | 0.00s | 1.2ms | 31.6ms |
| A_fast_decay | 15.50s | 18.3ms | 1.8ms | 0.00s | 0.00s | 0.00s | 1.1ms | 31.1ms |
| A_fast_decay | 13.18s | 16.4ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.2ms | 33.2ms |
| A_fast_decay | 14.57s | 20.8ms | 1.9ms | 0.00s | 0.00s | 0.00s | 1.1ms | 30.2ms |
| A_fast_decay | 14.60s | 26.1ms | 1.5ms | 0.00s | 0.00s | 0.00s | 1.5ms | 63.2ms |
| A_fast_decay | 14.88s | 24.2ms | 1.3ms | 0.00s | 0.00s | 0.00s | 1.6ms | 65.5ms |
| A_fast_decay | 14.15s | 24.5ms | 1.7ms | 0.00s | 0.00s | 0.00s | 1.3ms | 42.4ms |
| A_fast_decay | 15.02s | 24.8ms | 1.4ms | 0.00s | 0.00s | 0.00s | 1.5ms | 87.6ms |
| A_fast_decay | 14.28s | 21.1ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.3ms | 63.5ms |
| A_fast_decay | 14.31s | 26.8ms | 1.6ms | 0.00s | 0.00s | 0.00s | 1.5ms | 64.3ms |
| original_mrdbscp | 135.87s | 11.83s | 1.0ms | 0.00s | 0.00s | 0.00s | 1.2ms | 30.9ms |
| original_mrdbscp | 62.48s | 41.6ms | 1.6ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.3ms |
| original_mrdbscp | 61.82s | 44.0ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.6ms |
| original_mrdbscp | 62.04s | 43.1ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.1ms | 30.6ms |
| original_mrdbscp | 62.05s | 44.8ms | 1.1ms | 0.00s | 0.00s | 0.00s | 1.1ms | 48.7ms |
| original_mrdbscp | 61.89s | 43.1ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.1ms | 31.8ms |
| original_mrdbscp | 61.92s | 38.0ms | 1.3ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.1ms |
| original_mrdbscp | 62.07s | 39.0ms | 1.3ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.5ms |
| original_mrdbscp | 62.07s | 41.1ms | 0.9ms | 0.00s | 0.00s | 0.00s | 1.2ms | 51.2ms |
| original_mrdbscp | 61.96s | 43.5ms | 1.2ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.9ms |
| sinr_cps2 | 324.21s | 801.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 31.5ms |
| sinr_cps2 | 270.81s | 48.2ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 28.8ms |
| sinr_cps2 | 271.13s | 49.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 29.8ms |
| sinr_cps2 | 271.08s | 48.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 29.4ms |
| sinr_cps2 | 270.90s | 47.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 28.7ms |
| sinr_cps2 | 270.57s | 50.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 31.4ms |
| sinr_cps2 | 270.86s | 48.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 29.2ms |
| sinr_cps2 | 270.88s | 48.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 29.2ms |
| sinr_cps2 | 270.84s | 47.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 29.5ms |
| sinr_cps2 | 270.77s | 49.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.9ms |
| sinr_cps4 | 153.81s | 1.01s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 29.6ms |
| sinr_cps4 | 106.80s | 27.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 30.4ms |
| sinr_cps4 | 107.05s | 26.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 31.0ms |
| sinr_cps4 | 106.79s | 25.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 30.8ms |
| sinr_cps4 | 106.96s | 24.3ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 30.9ms |
| sinr_cps4 | 106.93s | 24.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 31.2ms |
| sinr_cps4 | 107.00s | 25.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 30.4ms |
| sinr_cps4 | 106.92s | 24.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 28.9ms |
| sinr_cps4 | 107.03s | 24.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 44.1ms |
| sinr_cps4 | 107.00s | 25.1ms | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 28.9ms |

## Worked Numeric Examples

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 27.62
  + initial_dense_grid/small setup 0.39
  + stage 1 pre 8.64
  + stage 1 optimizer 14.03
  + stage 1 post-to-DVF 0.07
  + stage 2 pre 3.16
  + stage 2 optimizer 21.49
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.37
  + stage 3 optimizer 68.75
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 144.55

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 144.55
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 144.63
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.50
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 4.44
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 3.00
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.06
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 16.02

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 16.02
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 16.09
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.37
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.56
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.56
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.06
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 13.56

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 13.56
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 13.66
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.59
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 3.11
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.87
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.99

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.99
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 15.08
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.63
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.63
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 3.11
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.86
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.26

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.26
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 15.38
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.74
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.73
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 9.40
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.44

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.44
  + dvf_file_write 0.07
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 15.56
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.59
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.73
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.79
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.63
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.76

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.76
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 14.84
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.60
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.46
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.41
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 10.15
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 15.64

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 15.64
  + dvf_file_write 0.09
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 15.78
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.67
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 2.98
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.75
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.56
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.97

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.97
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 15.10
```

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 0.66
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 3.07
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 2.66
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 8.58
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.01
  = 14.99

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 14.99
  + dvf_file_write 0.06
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 15.13
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 29.70
  + initial_dense_grid/small setup 0.02
  + stage 1 pre 7.30
  + stage 1 optimizer 13.51
  + stage 1 post-to-DVF 0.07
  + stage 2 pre 3.02
  + stage 2 optimizer 29.29
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.31
  + stage 3 optimizer 93.08
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 1.12
  = 177.41

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 177.41
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 177.48
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.33
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.90
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.55
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 42.03
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 62.84

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.84
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 62.90
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.57
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.31
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.94
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.03
  = 62.29

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.29
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 62.36
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.61
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.45
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.98
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 62.48

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.48
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 62.55
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.61
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.48
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.96
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.05
  = 62.53

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.53
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 62.61
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.35
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.59
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.33
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.96
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.05
  = 62.30

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.30
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 62.36
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.39
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.64
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.36
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.91
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 62.36

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.36
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.12
  = 62.50
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.46
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.61
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.47
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.99
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 62.58

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.58
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 62.65
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.66
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.45
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.96
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 62.51

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.51
  + dvf_file_write 0.05
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 62.60
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 0.34
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 6.61
  + stage 1 post-to-DVF 0.00
  + stage 2 pre 0.00
  + stage 2 optimizer 13.38
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.00
  + stage 3 optimizer 41.98
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.08
  = 62.39

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 62.39
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 62.45
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 28.78
  + initial_dense_grid/small setup 2.66
  + stage 1 pre 0.70
  + stage 1 optimizer 324.21
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 356.39

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 356.39
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.10
  = 356.53
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.81
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.29

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.29
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.36
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.44
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 271.13
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.04
  = 271.65

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.65
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.72
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.38
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 271.08
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.51

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.51
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.58
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.90
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.37

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.37
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.44
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.55
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.57
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.17

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.17
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 271.25
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.86
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.33

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.33
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.40
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.54
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.88
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.03
  = 271.48

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.48
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.55
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.84
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.34

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.34
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.41
```

```text
sinr_cps2 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 270.77
  + stage 1 post-to-DVF 0.03
  + other pre-DVF measured residual 0.02
  = 271.24

sinr_cps2 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 271.24
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.04
  = 271.31
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 26.89
  + initial_dense_grid/small setup 1.00
  + stage 1 pre 0.77
  + stage 1 optimizer 153.81
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 182.49

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 182.49
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.29
  = 182.81
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.80
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.28

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.28
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 107.37
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 107.05
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.49

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.49
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.08
  = 107.60
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.40
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.79
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.22

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.22
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 107.33
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.43
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.96
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.42

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.42
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 107.51
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.93
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.37

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.37
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 107.47
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.42
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 107.00
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.44

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.44
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 107.53
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.45
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 106.92
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.40

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.40
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 107.49
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.41
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 107.03
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.47

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.47
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.07
  = 107.59
```

```text
sinr_cps4 full_script_to_dvf_in_memory
  = setup 0.35
  + initial_dense_grid/small setup 0.00
  + stage 1 pre 0.00
  + stage 1 optimizer 107.00
  + stage 1 post-to-DVF 0.01
  + other pre-DVF measured residual 0.02
  = 107.38

sinr_cps4 full_script_to_dvf_file
  = full_script_to_dvf_in_memory 107.38
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.06
  = 107.47
```

## Direct Savings

Savings are `original_mrdbscp - A_fast_decay`; positive values mean `A_fast_decay` is faster.

| component | savings |
|---|---:|
| full_script_to_dvf_in_memory | 32.86s |
| loaded_images_to_dvf_in_memory | 30.78s |
| full_script_to_dvf_file | 32.85s |
| loaded_images_to_dvf_file | 30.76s |
| dvf_file_write | -0.7ms |
| external_wall | 32.85s |
| external_uninstrumented | 1.3ms |
| total_optimizer_loop | 31.60s |
| total_non_loop_training | -446.0ms |
| total_pure_knot_insertion | -0.5ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | 0.0ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 46.82s |
| loaded_images_to_dvf_in_memory | 46.99s |
| full_script_to_dvf_file | 46.81s |
| loaded_images_to_dvf_file | 46.98s |
| dvf_file_write | -1.8ms |
| external_wall | 46.82s |
| external_uninstrumented | 2.9ms |
| total_optimizer_loop | 46.97s |
| total_non_loop_training | 23.4ms |
| total_pure_knot_insertion | -0.2ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | 0.0ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 48.73s |
| loaded_images_to_dvf_in_memory | 48.66s |
| full_script_to_dvf_file | 48.70s |
| loaded_images_to_dvf_file | 48.63s |
| dvf_file_write | -3.6ms |
| external_wall | 48.70s |
| external_uninstrumented | -1.8ms |
| total_optimizer_loop | 48.64s |
| total_non_loop_training | 27.6ms |
| total_pure_knot_insertion | -0.0ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.0ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 47.48s |
| loaded_images_to_dvf_in_memory | 47.49s |
| full_script_to_dvf_file | 47.47s |
| loaded_images_to_dvf_file | 47.47s |
| dvf_file_write | 0.4ms |
| external_wall | 47.47s |
| external_uninstrumented | 1.4ms |
| total_optimizer_loop | 47.47s |
| total_non_loop_training | 22.3ms |
| total_pure_knot_insertion | -0.6ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | 0.0ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 47.27s |
| loaded_images_to_dvf_in_memory | 47.47s |
| full_script_to_dvf_file | 47.24s |
| loaded_images_to_dvf_file | 47.43s |
| dvf_file_write | -14.5ms |
| external_wall | 47.23s |
| external_uninstrumented | -1.8ms |
| total_optimizer_loop | 47.45s |
| total_non_loop_training | 18.7ms |
| total_pure_knot_insertion | -0.5ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.4ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 46.86s |
| loaded_images_to_dvf_in_memory | 47.03s |
| full_script_to_dvf_file | 46.81s |
| loaded_images_to_dvf_file | 46.98s |
| dvf_file_write | -33.6ms |
| external_wall | 46.81s |
| external_uninstrumented | 0.3ms |
| total_optimizer_loop | 47.01s |
| total_non_loop_training | 18.9ms |
| total_pure_knot_insertion | -0.0ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.4ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 47.59s |
| loaded_images_to_dvf_in_memory | 47.78s |
| full_script_to_dvf_file | 47.66s |
| loaded_images_to_dvf_file | 47.84s |
| dvf_file_write | -13.2ms |
| external_wall | 47.66s |
| external_uninstrumented | 0.9ms |
| total_optimizer_loop | 47.77s |
| total_non_loop_training | 13.5ms |
| total_pure_knot_insertion | -0.4ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.2ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 46.94s |
| loaded_images_to_dvf_in_memory | 47.07s |
| full_script_to_dvf_file | 46.87s |
| loaded_images_to_dvf_file | 47.00s |
| dvf_file_write | -58.2ms |
| external_wall | 46.87s |
| external_uninstrumented | 2.6ms |
| total_optimizer_loop | 47.06s |
| total_non_loop_training | 14.2ms |
| total_pure_knot_insertion | -0.1ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.4ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 47.54s |
| loaded_images_to_dvf_in_memory | 47.81s |
| full_script_to_dvf_file | 47.50s |
| loaded_images_to_dvf_file | 47.76s |
| dvf_file_write | -12.3ms |
| external_wall | 47.50s |
| external_uninstrumented | 0.3ms |
| total_optimizer_loop | 47.79s |
| total_non_loop_training | 20.0ms |
| total_pure_knot_insertion | -0.4ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.1ms |
| total_diagnostic_dense_overhead | 0.00s |
| full_script_to_dvf_in_memory | 47.40s |
| loaded_images_to_dvf_in_memory | 47.67s |
| full_script_to_dvf_file | 47.32s |
| loaded_images_to_dvf_file | 47.59s |
| dvf_file_write | -34.4ms |
| external_wall | 47.32s |
| external_uninstrumented | 0.1ms |
| total_optimizer_loop | 47.65s |
| total_non_loop_training | 16.8ms |
| total_pure_knot_insertion | -0.4ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | -0.4ms |
| total_diagnostic_dense_overhead | 0.00s |
| stage_1_pre_stage_setup | -1.34s |
| stage_1_optimizer_loop | -522.4ms |
| stage_1_pure_knot_insertion | -0.0ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -2.9ms |
| stage_1_post_stage_processing | -2.9ms |
| stage_2_pre_stage_setup | -139.4ms |
| stage_2_optimizer_loop | 7.80s |
| stage_2_pure_knot_insertion | -0.4ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.1ms |
| stage_2_post_stage_processing | -0.1ms |
| stage_3_pre_stage_setup | -54.6ms |
| stage_3_optimizer_loop | 24.33s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | 0.0ms |
| stage_3_post_stage_processing_to_dvf | 0.0ms |
| stage_3_post_stage_processing | 0.0ms |
| stage_1_pre_stage_setup | -0.0ms |
| stage_1_optimizer_loop | 2.46s |
| stage_1_pure_knot_insertion | -0.2ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.2ms |
| stage_1_post_stage_processing | -0.2ms |
| stage_2_pre_stage_setup | 0.3ms |
| stage_2_optimizer_loop | 10.54s |
| stage_2_pure_knot_insertion | 0.0ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | 0.1ms |
| stage_2_post_stage_processing | 0.1ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 33.97s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | 0.0ms |
| stage_3_post_stage_processing_to_dvf | 0.0ms |
| stage_3_post_stage_processing | -0.0ms |
| stage_1_pre_stage_setup | -0.0ms |
| stage_1_optimizer_loop | 4.02s |
| stage_1_pure_knot_insertion | 0.4ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | 0.6ms |
| stage_1_post_stage_processing | 0.6ms |
| stage_2_pre_stage_setup | 0.3ms |
| stage_2_optimizer_loop | 10.74s |
| stage_2_pure_knot_insertion | -0.4ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.4ms |
| stage_2_post_stage_processing | -0.4ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 33.88s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.0ms |
| stage_3_post_stage_processing_to_dvf | -0.0ms |
| stage_3_post_stage_processing | 0.0ms |
| stage_1_pre_stage_setup | 0.1ms |
| stage_1_optimizer_loop | 4.02s |
| stage_1_pure_knot_insertion | -0.2ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.2ms |
| stage_1_post_stage_processing | -0.2ms |
| stage_2_pre_stage_setup | -0.5ms |
| stage_2_optimizer_loop | 10.33s |
| stage_2_pure_knot_insertion | -0.5ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.5ms |
| stage_2_post_stage_processing | -0.5ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 33.12s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | 0.0ms |
| stage_3_post_stage_processing_to_dvf | 0.0ms |
| stage_3_post_stage_processing | 0.1ms |
| stage_1_pre_stage_setup | -0.5ms |
| stage_1_optimizer_loop | 3.98s |
| stage_1_pure_knot_insertion | -0.1ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.1ms |
| stage_1_post_stage_processing | -0.1ms |
| stage_2_pre_stage_setup | -0.3ms |
| stage_2_optimizer_loop | 10.37s |
| stage_2_pure_knot_insertion | -0.3ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.3ms |
| stage_2_post_stage_processing | -0.3ms |
| stage_3_pre_stage_setup | -0.2ms |
| stage_3_optimizer_loop | 33.11s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.4ms |
| stage_3_post_stage_processing_to_dvf | -0.4ms |
| stage_3_post_stage_processing | -0.7ms |
| stage_1_pre_stage_setup | -0.4ms |
| stage_1_optimizer_loop | 3.85s |
| stage_1_pure_knot_insertion | 0.2ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | 0.3ms |
| stage_1_post_stage_processing | 0.3ms |
| stage_2_pre_stage_setup | 0.1ms |
| stage_2_optimizer_loop | 10.60s |
| stage_2_pure_knot_insertion | -0.2ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | 0.0ms |
| stage_2_post_stage_processing | 0.0ms |
| stage_3_pre_stage_setup | -0.0ms |
| stage_3_optimizer_loop | 32.56s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.4ms |
| stage_3_post_stage_processing_to_dvf | -0.4ms |
| stage_3_post_stage_processing | -0.8ms |
| stage_1_pre_stage_setup | -0.5ms |
| stage_1_optimizer_loop | 3.91s |
| stage_1_pure_knot_insertion | -0.0ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.5ms |
| stage_1_post_stage_processing | -0.5ms |
| stage_2_pre_stage_setup | -0.2ms |
| stage_2_optimizer_loop | 10.57s |
| stage_2_pure_knot_insertion | -0.4ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.3ms |
| stage_2_post_stage_processing | -0.3ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 33.29s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.2ms |
| stage_3_post_stage_processing_to_dvf | -0.2ms |
| stage_3_post_stage_processing | -0.3ms |
| stage_1_pre_stage_setup | -0.7ms |
| stage_1_optimizer_loop | 4.15s |
| stage_1_pure_knot_insertion | 0.0ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.2ms |
| stage_1_post_stage_processing | -0.2ms |
| stage_2_pre_stage_setup | 0.0ms |
| stage_2_optimizer_loop | 11.07s |
| stage_2_pure_knot_insertion | -0.1ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | 0.2ms |
| stage_2_post_stage_processing | 0.2ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 31.84s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.4ms |
| stage_3_post_stage_processing_to_dvf | -0.4ms |
| stage_3_post_stage_processing | -0.7ms |
| stage_1_pre_stage_setup | -0.3ms |
| stage_1_optimizer_loop | 3.69s |
| stage_1_pure_knot_insertion | -0.2ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.3ms |
| stage_1_post_stage_processing | -0.3ms |
| stage_2_pre_stage_setup | -0.0ms |
| stage_2_optimizer_loop | 10.70s |
| stage_2_pure_knot_insertion | -0.2ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | 0.2ms |
| stage_2_post_stage_processing | 0.2ms |
| stage_3_pre_stage_setup | -0.0ms |
| stage_3_optimizer_loop | 33.40s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.1ms |
| stage_3_post_stage_processing_to_dvf | -0.1ms |
| stage_3_post_stage_processing | -0.2ms |
| stage_1_pre_stage_setup | -0.4ms |
| stage_1_optimizer_loop | 3.54s |
| stage_1_pure_knot_insertion | -0.1ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.3ms |
| stage_1_post_stage_processing | -0.3ms |
| stage_2_pre_stage_setup | -0.4ms |
| stage_2_optimizer_loop | 10.71s |
| stage_2_pure_knot_insertion | -0.3ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.2ms |
| stage_2_post_stage_processing | -0.2ms |
| stage_3_pre_stage_setup | -0.1ms |
| stage_3_optimizer_loop | 33.40s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | -0.4ms |
| stage_3_post_stage_processing_to_dvf | -0.4ms |
| stage_3_post_stage_processing | -0.8ms |

## Previous Conservative Timing Comparison

Positive values mean the new production path is faster than the previous conservative DVF-only timing.
Endpoint rows are the primary comparison. Component rows can shift because skipping earlier dense expansions also changes where CUDA warmup/autotune cost appears.

| candidate | metric | previous | production | previous - production |
|---|---|---:|---:|---:|
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 177.41s | -175.75s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 147.71s | -146.43s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 177.48s | -175.70s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 147.77s | -146.39s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 11.83s | -11.81s |
| original_mrdbscp | total_optimizer_loop | 1.26s | 135.87s | -134.61s |
| original_mrdbscp | external_wall | 1.79s | 177.49s | -175.70s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.84s | -61.18s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.51s | -61.24s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.90s | -61.13s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.58s | -61.19s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 41.6ms | -26.0ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 62.48s | -61.22s |
| original_mrdbscp | external_wall | 1.79s | 62.91s | -61.13s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.29s | -60.63s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 61.86s | -60.58s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.36s | -60.58s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 61.92s | -60.54s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 44.0ms | -28.3ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 61.82s | -60.56s |
| original_mrdbscp | external_wall | 1.79s | 62.37s | -60.58s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.48s | -60.81s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.08s | -60.81s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.55s | -60.78s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.15s | -60.77s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 43.1ms | -27.5ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 62.04s | -60.78s |
| original_mrdbscp | external_wall | 1.79s | 62.56s | -60.77s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.53s | -60.87s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.09s | -60.82s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.61s | -60.84s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.18s | -60.79s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 44.8ms | -29.1ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 62.05s | -60.80s |
| original_mrdbscp | external_wall | 1.79s | 62.62s | -60.83s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.30s | -60.63s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 61.93s | -60.65s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.36s | -60.59s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.00s | -60.61s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 43.1ms | -27.4ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 61.89s | -60.63s |
| original_mrdbscp | external_wall | 1.79s | 62.38s | -60.59s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.36s | -60.69s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 61.95s | -60.68s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.50s | -60.73s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.09s | -60.71s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 38.0ms | -22.3ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 61.92s | -60.66s |
| original_mrdbscp | external_wall | 1.79s | 62.51s | -60.72s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.58s | -60.92s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.11s | -60.83s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.65s | -60.87s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.17s | -60.79s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 39.0ms | -23.4ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 62.07s | -60.82s |
| original_mrdbscp | external_wall | 1.79s | 62.66s | -60.87s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.51s | -60.85s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.11s | -60.84s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.60s | -60.82s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.20s | -60.81s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 41.1ms | -25.5ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 62.07s | -60.82s |
| original_mrdbscp | external_wall | 1.79s | 62.61s | -60.82s |
| original_mrdbscp | full_script_to_dvf_in_memory | 1.66s | 62.39s | -60.73s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 1.27s | 62.00s | -60.73s |
| original_mrdbscp | full_script_to_dvf_file | 1.77s | 62.45s | -60.68s |
| original_mrdbscp | loaded_images_to_dvf_file | 1.38s | 62.06s | -60.68s |
| original_mrdbscp | total_non_loop_training | 15.7ms | 43.5ms | -27.9ms |
| original_mrdbscp | total_optimizer_loop | 1.26s | 61.96s | -60.70s |
| original_mrdbscp | external_wall | 1.79s | 62.46s | -60.68s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 144.55s | -141.42s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 116.93s | -114.31s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 144.63s | -141.44s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 117.01s | -114.34s |
| A_fast_decay | total_non_loop_training | 13.7ms | 12.27s | -12.26s |
| A_fast_decay | total_optimizer_loop | 2.60s | 104.27s | -101.67s |
| A_fast_decay | external_wall | 3.20s | 144.64s | -141.44s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 16.02s | -12.88s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 15.52s | -12.90s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 16.09s | -12.90s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 15.59s | -12.92s |
| A_fast_decay | total_non_loop_training | 13.7ms | 18.3ms | -4.6ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 15.50s | -12.90s |
| A_fast_decay | external_wall | 3.20s | 16.10s | -12.90s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 13.56s | -10.43s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 13.19s | -10.58s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 13.66s | -10.47s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 13.29s | -10.61s |
| A_fast_decay | total_non_loop_training | 13.7ms | 16.4ms | -2.7ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 13.18s | -10.58s |
| A_fast_decay | external_wall | 3.20s | 13.67s | -10.47s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 14.99s | -11.86s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.59s | -11.98s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.08s | -11.89s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 14.68s | -12.00s |
| A_fast_decay | total_non_loop_training | 13.7ms | 20.8ms | -7.1ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.57s | -11.97s |
| A_fast_decay | external_wall | 3.20s | 15.09s | -11.89s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 15.26s | -12.13s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.62s | -12.01s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.38s | -12.19s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 14.74s | -12.07s |
| A_fast_decay | total_non_loop_training | 13.7ms | 26.1ms | -12.4ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.60s | -12.00s |
| A_fast_decay | external_wall | 3.20s | 15.39s | -12.19s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 15.44s | -12.31s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.90s | -12.28s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.56s | -12.37s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 15.02s | -12.34s |
| A_fast_decay | total_non_loop_training | 13.7ms | 24.2ms | -10.5ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.88s | -12.28s |
| A_fast_decay | external_wall | 3.20s | 15.57s | -12.37s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 14.76s | -11.63s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.17s | -11.56s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 14.84s | -11.65s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 14.25s | -11.58s |
| A_fast_decay | total_non_loop_training | 13.7ms | 24.5ms | -10.8ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.15s | -11.55s |
| A_fast_decay | external_wall | 3.20s | 14.85s | -11.65s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 15.64s | -12.51s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 15.04s | -12.43s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.78s | -12.59s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 15.18s | -12.50s |
| A_fast_decay | total_non_loop_training | 13.7ms | 24.8ms | -11.1ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 15.02s | -12.42s |
| A_fast_decay | external_wall | 3.20s | 15.79s | -12.59s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 14.97s | -11.84s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.30s | -11.69s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.10s | -11.91s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 14.43s | -11.76s |
| A_fast_decay | total_non_loop_training | 13.7ms | 21.1ms | -7.4ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.28s | -11.68s |
| A_fast_decay | external_wall | 3.20s | 15.11s | -11.91s |
| A_fast_decay | full_script_to_dvf_in_memory | 3.13s | 14.99s | -11.86s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 2.61s | 14.33s | -11.72s |
| A_fast_decay | full_script_to_dvf_file | 3.19s | 15.13s | -11.94s |
| A_fast_decay | loaded_images_to_dvf_file | 2.67s | 14.47s | -11.80s |
| A_fast_decay | total_non_loop_training | 13.7ms | 26.8ms | -13.1ms |
| A_fast_decay | total_optimizer_loop | 2.60s | 14.31s | -11.71s |
| A_fast_decay | external_wall | 3.20s | 15.14s | -11.94s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 182.49s | -181.60s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 155.60s | -155.07s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 182.81s | -181.86s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 155.92s | -155.33s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 1.01s | -997.9ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 153.81s | -153.29s |
| sinr_cps4 | external_wall | 957.0ms | 182.81s | -181.86s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.28s | -106.39s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 106.82s | -106.29s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.37s | -106.43s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 106.92s | -106.33s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 27.7ms | -19.6ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 106.80s | -106.28s |
| sinr_cps4 | external_wall | 957.0ms | 107.38s | -106.42s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.49s | -106.60s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 107.07s | -106.54s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.60s | -106.65s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.18s | -106.59s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 26.0ms | -17.9ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 107.05s | -106.53s |
| sinr_cps4 | external_wall | 957.0ms | 107.61s | -106.65s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.22s | -106.34s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 106.82s | -106.29s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.33s | -106.38s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 106.92s | -106.34s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 25.8ms | -17.8ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 106.79s | -106.27s |
| sinr_cps4 | external_wall | 957.0ms | 107.33s | -106.37s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.42s | -106.53s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 106.99s | -106.46s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.51s | -106.56s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.08s | -106.49s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 24.3ms | -16.3ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 106.96s | -106.44s |
| sinr_cps4 | external_wall | 957.0ms | 107.52s | -106.56s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.37s | -106.48s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 106.96s | -106.43s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.47s | -106.52s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.06s | -106.47s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 24.6ms | -16.6ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 106.93s | -106.41s |
| sinr_cps4 | external_wall | 957.0ms | 107.47s | -106.52s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.44s | -106.56s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 107.03s | -106.50s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.53s | -106.58s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.11s | -106.52s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 25.5ms | -17.4ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 107.00s | -106.48s |
| sinr_cps4 | external_wall | 957.0ms | 107.53s | -106.58s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.40s | -106.51s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 106.95s | -106.42s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.49s | -106.55s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.05s | -106.46s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 24.7ms | -16.6ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 106.92s | -106.40s |
| sinr_cps4 | external_wall | 957.0ms | 107.50s | -106.54s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.47s | -106.59s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 107.06s | -106.53s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.59s | -106.64s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.17s | -106.58s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 24.1ms | -16.1ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 107.03s | -106.51s |
| sinr_cps4 | external_wall | 957.0ms | 107.59s | -106.64s |
| sinr_cps4 | full_script_to_dvf_in_memory | 886.8ms | 107.38s | -106.50s |
| sinr_cps4 | loaded_images_to_dvf_in_memory | 528.9ms | 107.03s | -106.50s |
| sinr_cps4 | full_script_to_dvf_file | 946.7ms | 107.47s | -106.53s |
| sinr_cps4 | loaded_images_to_dvf_file | 588.8ms | 107.12s | -106.53s |
| sinr_cps4 | total_non_loop_training | 8.1ms | 25.1ms | -17.1ms |
| sinr_cps4 | total_optimizer_loop | 520.0ms | 107.00s | -106.48s |
| sinr_cps4 | external_wall | 957.0ms | 107.48s | -106.52s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 356.39s | -355.92s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 327.62s | -327.48s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 356.53s | -355.99s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 327.75s | -327.56s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 801.7ms | -771.3ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 324.21s | -324.11s |
| sinr_cps2 | external_wall | 547.9ms | 356.54s | -355.99s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.29s | -270.81s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.86s | -270.73s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.36s | -270.82s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 270.93s | -270.73s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 48.2ms | -17.7ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.81s | -270.71s |
| sinr_cps2 | external_wall | 547.9ms | 271.36s | -270.81s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.65s | -271.17s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 271.18s | -271.05s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.72s | -271.18s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 271.25s | -271.06s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 49.1ms | -18.6ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 271.13s | -271.03s |
| sinr_cps2 | external_wall | 547.9ms | 271.73s | -271.18s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.51s | -271.03s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 271.12s | -270.99s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.58s | -271.04s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 271.20s | -271.00s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 48.3ms | -17.8ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 271.08s | -270.97s |
| sinr_cps2 | external_wall | 547.9ms | 271.59s | -271.04s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.37s | -270.90s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.94s | -270.81s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.44s | -270.91s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 271.01s | -270.82s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 47.9ms | -17.4ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.90s | -270.79s |
| sinr_cps2 | external_wall | 547.9ms | 271.45s | -270.90s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.17s | -270.70s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.62s | -270.49s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.25s | -270.71s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 270.70s | -270.50s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 50.6ms | -20.1ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.57s | -270.47s |
| sinr_cps2 | external_wall | 547.9ms | 271.26s | -270.71s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.33s | -270.85s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.90s | -270.77s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.40s | -270.86s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 270.97s | -270.78s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 48.0ms | -17.5ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.86s | -270.75s |
| sinr_cps2 | external_wall | 547.9ms | 271.41s | -270.86s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.48s | -271.00s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.93s | -270.79s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.55s | -271.01s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 271.00s | -270.80s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 48.8ms | -18.4ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.88s | -270.77s |
| sinr_cps2 | external_wall | 547.9ms | 271.56s | -271.01s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.34s | -270.86s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.89s | -270.75s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.41s | -270.87s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 270.96s | -270.76s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 47.9ms | -17.4ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.84s | -270.73s |
| sinr_cps2 | external_wall | 547.9ms | 271.41s | -270.87s |
| sinr_cps2 | full_script_to_dvf_in_memory | 477.0ms | 271.24s | -270.76s |
| sinr_cps2 | loaded_images_to_dvf_in_memory | 135.3ms | 270.82s | -270.69s |
| sinr_cps2 | full_script_to_dvf_file | 537.8ms | 271.31s | -270.78s |
| sinr_cps2 | loaded_images_to_dvf_file | 196.2ms | 270.89s | -270.70s |
| sinr_cps2 | total_non_loop_training | 30.5ms | 49.5ms | -19.0ms |
| sinr_cps2 | total_optimizer_loop | 104.1ms | 270.77s | -270.67s |
| sinr_cps2 | external_wall | 547.9ms | 271.32s | -270.77s |

## Stage Breakdown

| candidate | stage | pre | optimizer | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | post-to-DVF | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 1 | 8.64s | 14.03s | 0.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 69.1ms | 150.0 |
| A_fast_decay | 1 | 1.7ms | 4.44s | 1.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 150.0 |
| A_fast_decay | 1 | 1.8ms | 2.56s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.0ms | 150.0 |
| A_fast_decay | 1 | 1.7ms | 2.59s | 1.0ms | 0.00s | 0.00s | 0.00s | 0.00s | 2.0ms | 150.0 |
| A_fast_decay | 1 | 2.2ms | 2.63s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 150.0 |
| A_fast_decay | 1 | 2.2ms | 2.74s | 0.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 150.0 |
| A_fast_decay | 1 | 2.2ms | 2.73s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 150.0 |
| A_fast_decay | 1 | 2.5ms | 2.46s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 150.0 |
| A_fast_decay | 1 | 2.2ms | 2.98s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 150.0 |
| A_fast_decay | 1 | 2.2ms | 3.07s | 0.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.9ms | 150.0 |
| A_fast_decay | 2 | 3.16s | 21.49s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 200.0 |
| A_fast_decay | 2 | 1.3ms | 3.00s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 175.0 |
| A_fast_decay | 2 | 1.3ms | 2.56s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 150.0 |
| A_fast_decay | 2 | 2.0ms | 3.11s | 0.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.8ms | 175.0 |
| A_fast_decay | 2 | 1.5ms | 3.11s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 175.0 |
| A_fast_decay | 2 | 1.5ms | 2.73s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 175.0 |
| A_fast_decay | 2 | 1.9ms | 2.79s | 0.9ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 150.0 |
| A_fast_decay | 2 | 1.8ms | 2.41s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 150.0 |
| A_fast_decay | 2 | 1.8ms | 2.75s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 175.0 |
| A_fast_decay | 2 | 1.9ms | 2.66s | 0.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 150.0 |
| A_fast_decay | 3 | 368.9ms | 68.75s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 250.0 |
| A_fast_decay | 3 | 0.6ms | 8.06s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 250.0 |
| A_fast_decay | 3 | 0.6ms | 8.06s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 250.0 |
| A_fast_decay | 3 | 0.6ms | 8.87s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 275.0 |
| A_fast_decay | 3 | 0.6ms | 8.86s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 250.0 |
| A_fast_decay | 3 | 0.5ms | 9.40s | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 1.6ms | 275.0 |
| A_fast_decay | 3 | 0.6ms | 8.63s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 250.0 |
| A_fast_decay | 3 | 0.6ms | 10.15s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 300.0 |
| A_fast_decay | 3 | 0.5ms | 8.56s | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 1.3ms | 250.0 |
| A_fast_decay | 3 | 0.6ms | 8.58s | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 1.5ms | 250.0 |
| original_mrdbscp | 1 | 7.30s | 13.51s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 66.2ms | 400.0 |
| original_mrdbscp | 1 | 1.7ms | 6.90s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.5ms | 400.0 |
| original_mrdbscp | 1 | 1.8ms | 6.57s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 400.0 |
| original_mrdbscp | 1 | 1.8ms | 6.61s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 400.0 |
| original_mrdbscp | 1 | 1.7ms | 6.61s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 400.0 |
| original_mrdbscp | 1 | 1.8ms | 6.59s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 400.0 |
| original_mrdbscp | 1 | 1.7ms | 6.64s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 400.0 |
| original_mrdbscp | 1 | 1.8ms | 6.61s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 400.0 |
| original_mrdbscp | 1 | 1.9ms | 6.66s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.0ms | 400.0 |
| original_mrdbscp | 1 | 1.8ms | 6.61s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.6ms | 400.0 |
| original_mrdbscp | 2 | 3.02s | 29.29s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 2 | 1.6ms | 13.55s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 800.0 |
| original_mrdbscp | 2 | 1.6ms | 13.31s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 800.0 |
| original_mrdbscp | 2 | 1.5ms | 13.45s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 2 | 1.3ms | 13.48s | 0.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 2 | 1.6ms | 13.33s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 2 | 1.7ms | 13.36s | 0.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 2 | 1.8ms | 13.47s | 0.5ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 800.0 |
| original_mrdbscp | 2 | 1.8ms | 13.45s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.4ms | 800.0 |
| original_mrdbscp | 2 | 1.6ms | 13.38s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 3 | 314.2ms | 93.08s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 42.03s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.94s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.98s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.4ms | 41.96s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.96s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.91s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.99s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.96s | 0.00s | 0.00s | 0.00s | 0.00s | 1.2ms | 1.2ms | 1300.0 |
| original_mrdbscp | 3 | 0.5ms | 41.98s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |
| sinr_cps2 | 1 | 697.2ms | 324.21s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 2500.0 |
| sinr_cps2 | 1 | 1.9ms | 270.81s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 2500.0 |
| sinr_cps2 | 1 | 3.2ms | 271.13s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 2.0ms | 271.08s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 2.2ms | 270.90s | 0.00s | 0.00s | 0.00s | 0.00s | 27.9ms | 27.9ms | 2500.0 |
| sinr_cps2 | 1 | 2.4ms | 270.57s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 2.0ms | 270.86s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 2.2ms | 270.88s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 1.9ms | 270.84s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps2 | 1 | 2.1ms | 270.77s | 0.00s | 0.00s | 0.00s | 0.00s | 28.0ms | 28.0ms | 2500.0 |
| sinr_cps4 | 1 | 766.2ms | 153.81s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |
| sinr_cps4 | 1 | 2.1ms | 106.80s | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 5.5ms | 2500.0 |
| sinr_cps4 | 1 | 2.2ms | 107.05s | 0.00s | 0.00s | 0.00s | 0.00s | 5.5ms | 5.5ms | 2500.0 |
| sinr_cps4 | 1 | 1.9ms | 106.79s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 1.9ms | 106.96s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 1.9ms | 106.93s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 2.1ms | 107.00s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 1.8ms | 106.92s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 1.9ms | 107.03s | 0.00s | 0.00s | 0.00s | 0.00s | 5.3ms | 5.3ms | 2500.0 |
| sinr_cps4 | 1 | 2.1ms | 107.00s | 0.00s | 0.00s | 0.00s | 0.00s | 5.4ms | 5.4ms | 2500.0 |

## Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_first10.csv`
