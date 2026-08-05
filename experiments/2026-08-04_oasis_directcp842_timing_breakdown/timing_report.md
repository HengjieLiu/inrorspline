# DirectCP 8-4-2 Timing Breakdown

- Phase: `first10`
- Output root: `<external-run-root>/runs/oasis_directcp842_timing_breakdown_alpha1000_v1`
- Completed command rows: 20
- Timing JSON rows: 20/20
- Failed subprocesses: 0
- Non-target GPU compute-app observations: 0

## Reconciliation

- Residual failures: 0
- Parent intervals with large self-unaccounted time: 0
- External uninstrumented wall mean: 4.29s

## Timing Hierarchy And Definitions

Yes: `optimizer_loop_sec` is a strict subset of `training_elapsed_sec`.

But `known_fixed_or_io_sec` is not a single parent interval. It is a derived roll-up across several places in the run.

Conceptual hierarchy:

```text
external_wall
  script_total
    setup
    method_execution
      initial_dense_grid / small setup
      training
        stage/prolongation fixed work
        optimizer_loop
      post_train_evaluation
      small metric/model assembly
    sampler_save
    write_outputs
    other_script
  external_uninstrumented
```

Meaning of the major intervals:

- `external_wall_sec`: launcher-measured subprocess wall time, including Python startup/import, argument parsing, script execution, and process exit.
- `script_total_sec`: in-script wall time from `run_one.py` timing instrumentation. It is the reconciled parent for recorded script intervals.
- `method_execution_sec`: the method call itself; for DirectCP this includes training, post-training evaluation, and small model/metric assembly.
- `training_elapsed_sec`: DirectCP multistage training block only. It includes per-stage fixed work, prolongations, optimizer loops, and final dense expansions, but excludes post-training image/label evaluation and output writing.
- `optimizer_loop_sec`: strict subset of `training_elapsed_sec`; summed over the per-stage epoch loops.
- `known_fixed_or_io_sec`: derived roll-up, not a single parent interval. It sums known fixed stage work plus setup, post-training evaluation, output writing, and sampler save.
- `other_script_sec`: measured script time not assigned to `optimizer_loop_sec` or `known_fixed_or_io_sec`, such as initial dense grid setup, small model/metric assembly, and timing bookkeeping.
- `external_uninstrumented_sec`: `external_wall_sec - script_total_sec`; mostly process startup/import/argparse/exit and timing-artifact write overhead outside the script timer.

Important accounting relationships:

```text
external_wall_sec ~= script_total_sec + external_uninstrumented_sec
script_total_sec ~= optimizer_loop_sec + known_fixed_or_io_sec + other_script_sec
training_elapsed_sec ~= optimizer_loop_sec + non_loop_training_work
```

Because `known_fixed_or_io_sec` mixes intervals inside and outside training, it should not be added to `training_elapsed_sec`. It is designed to explain `script_total_sec` together with `optimizer_loop_sec` and `other_script_sec`.

## What Is In Fixed/IO Known

Defined in `scripts/experiments/oasis_alpha_sweep/analyze_directcp842_timing_breakdown.py` around line 204 as:

```text
fixed/io known =
  fixed_stage_known
  + setup
  + post_train_evaluation
  + write_outputs
  + sampler_save

fixed_stage_known = sum over stages:
  loss_volume_setup
  + stage_setup
  + prolongation
  + final_dense_expand
  + control_summary_stats
```

So `fixed/io known` includes:

- CUDA/device setup, random seeds, case load, case validation/print, and sampler initialization.
- Per-stage loss-volume creation.
- Per-stage grid/control/optimizer setup.
- Stage-to-stage prolongation.
- Final dense expansion after each stage.
- Control summary statistics.
- Post-training image warp, segmentation warp, Dice, HD95/ASD, and dense NCC.
- Regularity metrics.
- Dense displacement save.
- Model/control saves.
- CSV/JSON/summary/DONE writing.

It does not include:

- Optimizer-loop epochs.
- Small residual script work, reported as `other_script_sec`.
- Python process startup/import/argument parsing/process exit, reported as `external_uninstrumented_sec`.
- The `timing.json` write itself, which is effectively outside `script_total_sec`.

## Worked Accounting Examples

original:

```text
script_total 243.47
  = optimizer_loop 110.59
  + fixed/io known 132.10
  + other_script 0.78

wall 247.76
  = script_total 243.47
  + external_uninstrumented 4.29

training 206.12
  = optimizer_loop 110.59
  + non_loop_training_work 95.53
```

A_fast_decay:

```text
script_total 202.32
  = optimizer_loop 63.69
  + fixed/io known 137.80
  + other_script 0.82

wall 206.62
  = script_total 202.32
  + external_uninstrumented 4.30

training 161.79
  = optimizer_loop 63.69
  + non_loop_training_work 98.09
```

The key reason epoch reduction does not translate linearly is:

```text
A_fast_decay removes about 46.89s from optimizer_loop,
but the A_fast_decay fixed/io known bucket is still about 137.80s per case.
That is why a 1922.5 average epoch reduction gives only 41.14s wall saving per case.
```

## Method Means

| candidate | n | wall | script | training | optimizer loop | fixed/io known | epochs | Dice | HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 10 | 206.62s | 202.32s | 161.79s | 63.69s | 137.80s | 577.5 | 0.80581 | 2.2147 |
| original_mrdbscp | 10 | 247.76s | 243.47s | 206.12s | 110.59s | 132.10s | 2500.0 | 0.80697 | 2.2020 |

## Accounting Means

| candidate | wall | script | external uninstrumented | optimizer loop | fixed/io known | other script | training | training non-loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 206.62s | 202.32s | 4.30s | 63.69s | 137.80s | 0.82s | 161.79s | 98.09s |
| original_mrdbscp | 247.76s | 243.47s | 4.29s | 110.59s | 132.10s | 0.78s | 206.12s | 95.53s |

## Fixed/IO Known Components

`fixed/io known` is computed as:

```text
fixed/io known = fixed_stage_known + setup + post_train_evaluation + write_outputs + sampler_save
fixed_stage_known = sum_stages(loss_volume_setup + stage_setup + prolongation + final_dense_expand + control_summary_stats)
```

| candidate | fixed stage known | setup | post-train eval | write outputs | sampler save | fixed/io known |
|---|---:|---:|---:|---:|---:|---:|
| A_fast_decay | 97.62s | 28.06s | 5.25s | 6.88s | 0.00s | 137.80s |
| original_mrdbscp | 95.08s | 27.09s | 4.84s | 5.08s | 0.00s | 132.10s |

## Original vs A_fast_decay

| n pairs | wall savings | wall speedup | optimizer-loop savings | epoch reduction | Dice delta | HD95 delta |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 41.14s | 1.207x | 46.89s | 1922.5 | -0.00116 | 0.0127 |

## Stage Linear Fits

| stage | elapsed | n | intercept | slope/epoch | R2 |
|---:|---|---:|---:|---:|---:|
| 0 | stage_total_sec | 20 | 41.16s | 0.02s | 0.168 |
| 0 | optimizer_loop_sec | 20 | 9.45s | 0.02s | 0.652 |
| 1 | stage_total_sec | 20 | 21.74s | 0.02s | 0.834 |
| 1 | optimizer_loop_sec | 20 | 18.74s | 0.02s | 0.820 |
| 2 | stage_total_sec | 20 | 22.25s | 0.03s | 0.961 |
| 2 | optimizer_loop_sec | 20 | 22.24s | 0.03s | 0.961 |

## Interpretation

Epoch reduction only converts into wall-clock savings through the optimizer-loop bucket. Loss-volume setup, grid/control setup, prolongation, final dense expansion, dense image/label evaluation, regularity metrics, and serialization remain largely fixed per case. The `known_fixed_or_io_sec` column is therefore the main explanation when a large epoch reduction produces a modest total-wall reduction.

## Artifacts

- `timing_reconciliation.csv`
- `timing_stage_breakdown.csv`
- `timing_method_comparison.csv`
- `timing_stage_linear_fits.csv`
- `command_timing_<phase>.csv`
