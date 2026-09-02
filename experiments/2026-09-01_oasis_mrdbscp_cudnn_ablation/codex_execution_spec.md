# OASIS MR-D-BSCP cuDNN Cold/Warm Ablation

## Purpose

Measure whether the large first-case runtime overhead of the paper-facing
MR-D-BSCP fast schedule is caused by cuDNN benchmark selection,
cuDNN-deterministic algorithm constraints, or other first-use GPU behavior.

## Frozen protocol

- Candidate: MR_A_fast_decay.
- Method: direct_cp_8_4_2_be_bspline_analytic_multistage.
- Alpha: 1000.
- Seed: 1.
- Split: configs/splits/oasis/pair_test_200_shuffled_seed000003.txt.
- Case ranks: 0-4, in the listed seed-3 shuffled order.
- Physical GPU: 2; logical device inside the masked process is cuda:0.
- One fresh worker process per configuration.
- Within each worker: rank 0 is cold, then ranks 1-4 run sequentially as warm cases.
- Timing mode: dvf_only with deep phase timing.
- Timing endpoint: case_load_complete through dvf_file_ready; evaluation is excluded.

## Configuration matrix

| Configuration | cudnn.benchmark | cudnn.deterministic |
|---|---:|---:|
| benchmark_true_deterministic_false | true | false |
| benchmark_false_deterministic_false | false | false |
| benchmark_true_deterministic_true | true | true |
| benchmark_false_deterministic_true | false | true |

The four configurations are launched sequentially. No configuration is
silently replaced if deterministic CUDA operations are unsupported.

## Commands

All commands run inside `py310sitreg_v1_test1` using the documented
`/workspace/repo` repository mount. The controller command is:

    python /workspace/repo/scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py \
      --gpu 2 --case-ranks 0-4 --timing-detail deep

The controller writes raw timing outputs below
`/workspace/scratch/runs/oasis_mrdbscp_cudnn_ablation_20260901_v1` and
review summaries below this experiment directory.

## Analysis

For each configuration, report cold time, warm mean and standard deviation,
cold-minus-warm gap, setup.device time, training time, B-spline expansion,
backward/optimizer time, loss-volume setup, residual phase time, DVF output
time, actual epochs, peak memory, and effective backend metadata.

After all timing summaries pass, evaluate all 20 saved dense DVFs for Dice in a
separate sequential GPU-2 process. Dice evaluation is not part of the timing
endpoint.

## Conditional follow-up

No profiler is included in this batch. If disabling benchmark changes the
cold-minus-warm endpoint gap by less than approximately 20 percent, plan a
separate one-cold/one-warm profiler run to distinguish cuDNN algorithm selection
from CUDA kernel, allocator, or other first-use overhead.
