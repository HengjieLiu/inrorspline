# DirectCP 8-4-2 DVF-Only Runtime Timing

- Phase: `rank0`
- Output root: `<external-run-root>/runs/oasis_directcp842_production_dvf_timing_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-04_oasis_directcp842_production_dvf_timing`
- Completed command rows: 2
- Timing JSON rows: 2/2
- Dense DVF files present: 2/2
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
- `training`: the multistage optimization block, including non-loop stage work.
- `non_loop_training`: stage setup, pure knot insertion, optional diagnostic dense checks/saves, final output dense expansion, control stats, and other non-epoch work.
- `external_uninstrumented`: process startup/import/argparse/exit and timing-artifact overhead outside `script_total`.

Inclusion/exclusion rules:

- DVF-only headline times include setup, image/case load for full-script starts, DirectCP multistage optimization, final dense expansion, and optionally dense DVF file write.
- DVF-only headline times do not include image warp evaluation, segmentation warp, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, or full CSV packaging.
- Previous `fixed/io known` is not the DVF-only headline because it included evaluation and regularity/output packaging that are intentionally skipped here.
- Intermediate dense expansion is not required to start the next stage. Production DVF timing skips it by default.
- `pure_knot_insertion` means control-grid knot insertion only. Optional preservation dense expansion and metrics are reported separately.
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
| A_fast_decay | 129.29s | 101.27s | 129.36s | 101.33s | 34.2ms | 132.00s | 2.64s | 600.0 |
| original_mrdbscp | 174.72s | 149.83s | 174.78s | 149.89s | 29.8ms | 177.85s | 3.07s | 2500.0 |

## Grouped Totals

| candidate | optimizer loop | non-loop training | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | DVF file write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 89.87s | 10.60s | 1.4ms | 0.00s | 0.00s | 0.00s | 1.1ms | 34.2ms |
| original_mrdbscp | 139.44s | 10.39s | 1.1ms | 0.00s | 0.00s | 0.00s | 1.1ms | 29.8ms |

## Worked Numeric Examples

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 28.03
  + initial_dense_grid/small setup 0.79
  + stage 1 pre 6.26
  + stage 1 optimizer 11.58
  + stage 1 post-to-DVF 0.06
  + stage 2 pre 3.37
  + stage 2 optimizer 20.59
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.08
  + stage 3 optimizer 57.70
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.82
  = 129.29

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 129.29
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 129.36
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 24.89
  + initial_dense_grid/small setup 0.01
  + stage 1 pre 6.07
  + stage 1 optimizer 15.18
  + stage 1 post-to-DVF 0.06
  + stage 2 pre 2.41
  + stage 2 optimizer 31.99
  + stage 2 post-to-DVF 0.00
  + stage 3 pre 0.75
  + stage 3 optimizer 92.27
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 1.09
  = 174.72

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 174.72
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 174.78
```

## Direct Savings

Savings are `original_mrdbscp - A_fast_decay`; positive values mean `A_fast_decay` is faster.

| component | savings |
|---|---:|
| full_script_to_dvf_in_memory | 45.42s |
| loaded_images_to_dvf_in_memory | 48.56s |
| full_script_to_dvf_file | 45.42s |
| loaded_images_to_dvf_file | 48.56s |
| dvf_file_write | -4.4ms |
| external_wall | 45.84s |
| external_uninstrumented | 423.6ms |
| total_optimizer_loop | 49.56s |
| total_non_loop_training | -215.6ms |
| total_pure_knot_insertion | -0.3ms |
| total_prolongation_preserve_dense_expand | 0.00s |
| total_prolongation_preserve_metrics | 0.00s |
| total_intermediate_dense_expand | 0.00s |
| total_final_output_dense_expand | 0.0ms |
| total_diagnostic_dense_overhead | 0.00s |
| stage_1_pre_stage_setup | -188.7ms |
| stage_1_optimizer_loop | 3.60s |
| stage_1_pure_knot_insertion | 0.0ms |
| stage_1_prolongation_preserve_dense_expand | 0.00s |
| stage_1_prolongation_preserve_metrics | 0.00s |
| stage_1_intermediate_dense_expand | 0.00s |
| stage_1_final_output_dense_expand | 0.00s |
| stage_1_post_stage_processing_to_dvf | -0.9ms |
| stage_1_post_stage_processing | -0.9ms |
| stage_2_pre_stage_setup | -967.3ms |
| stage_2_optimizer_loop | 11.40s |
| stage_2_pure_knot_insertion | -0.4ms |
| stage_2_prolongation_preserve_dense_expand | 0.00s |
| stage_2_prolongation_preserve_metrics | 0.00s |
| stage_2_intermediate_dense_expand | 0.00s |
| stage_2_final_output_dense_expand | 0.00s |
| stage_2_post_stage_processing_to_dvf | -0.4ms |
| stage_2_post_stage_processing | -0.4ms |
| stage_3_pre_stage_setup | 670.3ms |
| stage_3_optimizer_loop | 34.56s |
| stage_3_pure_knot_insertion | 0.00s |
| stage_3_prolongation_preserve_dense_expand | 0.00s |
| stage_3_prolongation_preserve_metrics | 0.00s |
| stage_3_intermediate_dense_expand | 0.00s |
| stage_3_final_output_dense_expand | 0.0ms |
| stage_3_post_stage_processing_to_dvf | 0.0ms |
| stage_3_post_stage_processing | 0.1ms |

## Previous Conservative Timing Comparison

Positive values mean the new production path is faster than the previous conservative DVF-only timing.
Endpoint rows are the primary comparison. Component rows can shift because skipping earlier dense expansions also changes where CUDA warmup/autotune cost appears.

| candidate | metric | previous | production | previous - production |
|---|---|---:|---:|---:|
| original_mrdbscp | full_script_to_dvf_in_memory | 234.18s | 174.72s | 59.46s |
| original_mrdbscp | loaded_images_to_dvf_in_memory | 206.51s | 149.83s | 56.69s |
| original_mrdbscp | full_script_to_dvf_file | 234.26s | 174.78s | 59.48s |
| original_mrdbscp | loaded_images_to_dvf_file | 206.60s | 149.89s | 56.71s |
| original_mrdbscp | total_non_loop_training | 97.99s | 10.39s | 87.60s |
| original_mrdbscp | total_optimizer_loop | 108.51s | 139.44s | -30.92s |
| original_mrdbscp | external_wall | 238.22s | 177.85s | 60.38s |
| A_fast_decay | full_script_to_dvf_in_memory | 175.22s | 129.29s | 45.93s |
| A_fast_decay | loaded_images_to_dvf_in_memory | 146.77s | 101.27s | 45.50s |
| A_fast_decay | full_script_to_dvf_file | 175.28s | 129.36s | 45.92s |
| A_fast_decay | loaded_images_to_dvf_file | 146.83s | 101.33s | 45.50s |
| A_fast_decay | total_non_loop_training | 89.58s | 10.60s | 78.98s |
| A_fast_decay | total_optimizer_loop | 56.31s | 89.87s | -33.56s |
| A_fast_decay | external_wall | 178.99s | 132.00s | 46.98s |

## Stage Breakdown

| candidate | stage | pre | optimizer | pure knot | preserve dense | preserve metrics | intermediate dense | final output dense | post-to-DVF | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 1 | 6.26s | 11.58s | 0.6ms | 0.00s | 0.00s | 0.00s | 0.00s | 63.1ms | 150.0 |
| A_fast_decay | 2 | 3.37s | 20.59s | 0.8ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.7ms | 200.0 |
| A_fast_decay | 3 | 81.6ms | 57.70s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 250.0 |
| original_mrdbscp | 1 | 6.07s | 15.18s | 0.7ms | 0.00s | 0.00s | 0.00s | 0.00s | 62.2ms | 400.0 |
| original_mrdbscp | 2 | 2.41s | 31.99s | 0.4ms | 0.00s | 0.00s | 0.00s | 0.00s | 1.3ms | 800.0 |
| original_mrdbscp | 3 | 751.9ms | 92.27s | 0.00s | 0.00s | 0.00s | 0.00s | 1.1ms | 1.1ms | 1300.0 |

## Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_rank0.csv`
