---
doc_type: codex_execution_spec
status: complete
---

# GPU-3 OASIS Table-1 Candidates cuDNN Timing Rerun

## Purpose

Rerun every fast candidate represented in the camera-ready OASIS Table 1 on
the seed-3 shuffled first10 protocol, with a paired three-arm cuDNN backend
comparison. This produces a review report only; the camera-ready PDF and its
LaTeX source are out of scope until the results have been reviewed.

## Frozen protocol

- Candidates, in Table-1 order: `MR_A_fast_decay`, `D2_A_bal_lr10x`,
  `D4_A_bal_lr10x`, `S2_A_bal_lr10x`, `S4_A_bal_lr10x`.
- Methods: the repository candidate definitions in
  `scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py`.
- Alpha: `1000`; seed: `1`.
- OASIS root:
  `/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9`, in the listed seed-3 shuffled order.
- Physical GPU: `3`; the process masks it to logical `cuda:0`.
- Container: `py310sitreg_v1_artifact_store`.
- Image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Timing detail: `deep`; CPU thread limit: `6`.

The selected schedules are unchanged from the Table-1 fast candidates. In
particular, `MR_A_fast_decay` uses control points `8,4,2`, image downsampling
`4,2,1`, maximum stages `400,800,1300`, stage learning rates
`3e-2,1e-2,3e-3`, and adaptive stopping.

## Configuration and sequence

Each candidate runs these configurations strictly in order:

1. `benchmark=False, deterministic=True`
2. `benchmark=False, deterministic=False`
3. `benchmark=True, deterministic=False`

The execution order is candidate-major: for each candidate in Table-1 order,
run all three configurations in the order above. Each configuration gets a
fresh persistent worker. The worker measures rank `0` cold, then rank `0`
again and ranks `1-9` as ten warm measurements. The repeated rank-0 run has a
unique output directory.

Timing is `case_load_complete` through `dvf_file_ready`: it includes
optimization and dense-DVF preparation/writing, and excludes metric
evaluation.

## Expected outputs and acceptance

- Timing: `5 * 3 * 11 = 165` rows, all complete.
- Dice: one saved DVF per unique candidate/configuration/case,
  `5 * 3 * 10 = 150` rows, evaluated in a separate sequential GPU-3 process
  using fixed `benchmark=False, deterministic=True`; evaluation time is not
  part of timing.
- Every timing JSON records requested/effective backend flags.
- Every selected DVF is finite `float32`, normalized-grid XYZ, and has shape
  `(160, 224, 192, 3)`.
- The report records cold time, warm mean and sample SD, gap, setup-device
  time, expansion/backward gaps, actual steps, peak memory, backend metadata,
  and absolute/percentage comparisons to each candidate's paired reference.
- The report records the exact Git hash and dirty-tree state.

## Artifact locations

- Run root: `/database/inrorspline/runs/oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1`.
- Review root: `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3`.
- Report: `oasis_table1_candidates_cudnn_report.md` under the review root.

## Commands

All commands run inside the default container from
`/homebase/code_sync/inrorspline`, with `ARTIFACT_ROOT=/database/inrorspline`.
The full command uses the same arguments as the dry run below, without
`--dry-run` and without `--skip-dice`.

    python scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py \
      --gpu 3 \
      --container-name py310sitreg_v1_artifact_store \
      --oasis-root /mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1 \
      --case-ranks 0-9 \
      --candidates MR_A_fast_decay,D2_A_bal_lr10x,D4_A_bal_lr10x,S2_A_bal_lr10x,S4_A_bal_lr10x \
      --configs benchmark_false_deterministic_true,benchmark_false_deterministic_false,benchmark_true_deterministic_false \
      --repeat-cold-rank \
      --variant oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1 \
      --out-root /database/inrorspline/runs/oasis_table1_candidates_cudnn_first10_gpu3_20260916_v1 \
      --review-root /database/inrorspline/repo_experiments/2026-09-16_oasis_table1_candidates_cudnn_first10_gpu3 \
      --dry-run

The smoke command adds `--smoke --skip-dice` and uses separate `_smoke`
artifact roots. Smoke is run sequentially for all 15 candidate/configuration
pairs before the full timing launch.
