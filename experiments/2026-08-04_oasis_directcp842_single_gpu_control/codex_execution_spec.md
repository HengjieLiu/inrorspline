# Codex Execution Spec: OASIS DirectCP 8-4-2 Single-GPU Runtime Control

## Objective

Measure the original MR-D-BSCP / DirectCP `8-4-2` reference and `A_fast_decay`
under a controlled serial condition: one rank-0 OASIS job on GPU `0`, one job
at a time, with the other GPUs idle.

## Scope

- Dataset: OASIS BrainMR seed-3 shuffled rank `0`.
- Pair: fixed `0333`, moving `0115`.
- Original sorted case index: `158`.
- Split file:
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Method family: `direct_cp_8_4_2_be_bspline_analytic_multistage`.
- Stage cps: `8,4,2`.
- Stage downsample: `4,2,1`.
- Alpha: `1000`.
- Seed: `1`.
- Evaluation interval: `250`.
- Docker container: `py310sitreg_v1_test1`.
- GPU policy: GPU `0` only, serial execution.

## Methods

### Original MR-D-BSCP / DirectCP 8-4-2

- LR: `1e-3`.
- Stage epochs: `400,800,1300`.
- Early stopping: none.
- Runtime candidate tag: `single_gpu_original_mrdbscp`.

### A_fast_decay

- Base LR: `1e-2`.
- Stage LR: `3e-2,1e-2,3e-3`.
- Max stage epochs: `400,800,1300`.
- Early stopping: adaptive.
- Stage minimum epochs: `75,150,250`.
- Stage patience: `50,75,100`.
- Check interval: `25`.
- Relative plateau thresholds: total loss `1e-4`, data loss `5e-5`.
- Runtime candidate tag: `single_gpu_A_fast_decay`.

## Outputs

- External run root:
  `<external-run-root>/runs/oasis_directcp842_single_gpu_control_alpha1000_v1`.
- Review root:
  `experiments/2026-08-04_oasis_directcp842_single_gpu_control/`.

## Acceptance

- No GPU compute processes are active before launch.
- Baseline and candidate run sequentially on GPU `0`.
- Both runs write `summary.json` and `DONE`.
- Compare training time, total time, wall-clock command time, actual epochs,
  Dice, and HD95.
