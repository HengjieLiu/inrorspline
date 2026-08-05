# DirectCP 8-4-2 DVF-Only Runtime Timing

- Phase: `rank0`
- Output root: `<external-run-root>/runs/oasis_directcp842_dvf_only_timing_alpha1000_v1`
- Review root: `<repo-root>/experiments/2026-08-04_oasis_directcp842_dvf_only_timing`
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
- `non_loop_training`: stage setup, prolongation/knot insertion, final dense expansion, control stats, and other non-epoch work.
- `external_uninstrumented`: process startup/import/argparse/exit and timing-artifact overhead outside `script_total`.

Inclusion/exclusion rules:

- DVF-only headline times include setup, image/case load for full-script starts, DirectCP multistage optimization, final dense expansion, and optionally dense DVF file write.
- DVF-only headline times do not include image warp evaluation, segmentation warp, Dice, HD95/ASD, dense NCC, regularity metrics, model/control checkpoint saves, or full CSV packaging.
- Previous `fixed/io known` is not the DVF-only headline because it included evaluation and regularity/output packaging that are intentionally skipped here.
- The final-stage `dvf_in_memory_ready` mark is recorded immediately after final dense expansion. Final control-summary stats may occur after that mark, so they are part of total `non_loop_training` but not part of the in-memory DVF endpoint.

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
| A_fast_decay | 175.22s | 146.77s | 175.28s | 146.83s | 0.03s | 178.99s | 3.70s | 600.0 |
| original_mrdbscp | 234.18s | 206.51s | 234.26s | 206.60s | 0.04s | 238.22s | 3.96s | 2500.0 |

## Grouped Totals

| candidate | optimizer loop | non-loop training | pre-stage setup | post-stage processing | knot insertion | final dense expansion | DVF file write |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 56.31s | 89.58s | 7.46s | 82.08s | 57.98s | 24.04s | 0.03s |
| original_mrdbscp | 108.51s | 97.99s | 9.54s | 88.38s | 61.55s | 26.54s | 0.04s |

## Worked Numeric Examples

```text
A_fast_decay full_script_to_dvf_in_memory
  = setup 28.45
  + initial_dense_grid/small setup 0.88
  + stage 1 pre 6.07
  + stage 1 optimizer 9.69
  + stage 1 post-to-DVF 47.34
  + stage 2 pre 1.38
  + stage 2 optimizer 18.98
  + stage 2 post-to-DVF 34.74
  + stage 3 pre 0.00
  + stage 3 optimizer 27.64
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.04
  = 175.22

A_fast_decay full_script_to_dvf_file
  = full_script_to_dvf_in_memory 175.22
  + dvf_file_write 0.03
  + small file-prep/post-DVF overhead before file-ready 0.03
  = 175.28
```

```text
original_mrdbscp full_script_to_dvf_in_memory
  = setup 27.67
  + initial_dense_grid/small setup 0.02
  + stage 1 pre 6.48
  + stage 1 optimizer 14.06
  + stage 1 post-to-DVF 53.78
  + stage 2 pre 3.06
  + stage 2 optimizer 30.65
  + stage 2 post-to-DVF 34.59
  + stage 3 pre 0.00
  + stage 3 optimizer 63.80
  + stage 3 post-to-DVF 0.00
  + other pre-DVF measured residual 0.07
  = 234.18

original_mrdbscp full_script_to_dvf_file
  = full_script_to_dvf_in_memory 234.18
  + dvf_file_write 0.04
  + small file-prep/post-DVF overhead before file-ready 0.05
  = 234.26
```

## Direct Savings

Savings are `original_mrdbscp - A_fast_decay`; positive values mean `A_fast_decay` is faster.

| component | savings |
|---|---:|
| full_script_to_dvf_in_memory | 58.96s |
| loaded_images_to_dvf_in_memory | 59.75s |
| full_script_to_dvf_file | 58.98s |
| loaded_images_to_dvf_file | 59.77s |
| dvf_file_write | 0.00s |
| external_wall | 59.24s |
| external_uninstrumented | 0.26s |
| total_optimizer_loop | 52.20s |
| total_non_loop_training | 8.41s |
| stage_1_pre_stage_setup | 0.40s |
| stage_1_optimizer_loop | 4.36s |
| stage_1_post_stage_processing_to_dvf | 6.44s |
| stage_1_post_stage_processing | 6.44s |
| stage_2_pre_stage_setup | 1.67s |
| stage_2_optimizer_loop | 11.67s |
| stage_2_post_stage_processing_to_dvf | -0.15s |
| stage_2_post_stage_processing | -0.15s |
| stage_3_pre_stage_setup | -0.00s |
| stage_3_optimizer_loop | 36.17s |
| stage_3_post_stage_processing_to_dvf | 0.00s |
| stage_3_post_stage_processing | 0.00s |

## Stage Breakdown

| candidate | stage | pre | optimizer | post-to-DVF | post total | knot insertion | final dense | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 1 | 6.07s | 9.69s | 47.34s | 47.34s | 24.64s | 22.64s | 150.0 |
| A_fast_decay | 2 | 1.38s | 18.98s | 34.74s | 34.74s | 33.34s | 1.40s | 200.0 |
| A_fast_decay | 3 | 0.00s | 27.64s | 0.00s | 0.00s | 0.00s | 0.00s | 250.0 |
| original_mrdbscp | 1 | 6.48s | 14.06s | 53.78s | 53.78s | 27.44s | 26.05s | 400.0 |
| original_mrdbscp | 2 | 3.06s | 30.65s | 34.59s | 34.59s | 34.11s | 0.48s | 800.0 |
| original_mrdbscp | 3 | 0.00s | 63.80s | 0.00s | 0.00s | 0.00s | 0.00s | 1300.0 |

## Artifacts

- `dvf_only_timing_report.md`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `dvf_only_grouped_breakdown.csv`
- `dvf_only_reconciliation.csv`
- `command_timing_dvf_only_rank0.csv`
