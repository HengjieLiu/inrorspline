---
doc_type: codex_execution_spec
status: in_progress
---

# OASIS Table-1 Base/Fast F/T Rerun on GPU 3

## Purpose

Rerun the five 2500-step Table-1 base schedules and five selected fast
schedules under `benchmark=False, deterministic=True` using the seed-3
shuffled first-10 OASIS protocol. This is a diagnostic rerun for reviewing the
Table-1 backend setting; it is not a replacement for the paper's OASIS-100
accuracy study.

## Frozen protocol

- Alpha: `1000`; optimizer seed: `1`.
- OASIS root:
  `/mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1`.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9` in the seed-3 shuffled order.
- Physical GPU: `3`; the process masks it to logical `cuda:0`.
- Container: `py310sitreg_v1_artifact_store`.
- Image: `hengjieliu96/dl_dir:py310sitreg_v1`.
- Timing detail: `deep`; CPU thread limit: `6`.
- Backend: `benchmark=False, deterministic=True`.

The MR-D-BSCP fast schedule remains `MR_A_fast_decay`: control points
`8,4,2`, image downsampling `4,2,1`, maximum stages `400,800,1300`, stage
learning rates `3e-2,1e-2,3e-3`, and adaptive stopping.

## Candidate order

Candidates run sequentially in paired Table-1 order:

1. `MR_baseline_lr1x_full`, `MR_A_fast_decay`
2. `D2_baseline_lr1x_full`, `D2_A_bal_lr10x`
3. `D4_baseline_lr1x_full`, `D4_A_bal_lr10x`
4. `S2_baseline_lr1x_full`, `S2_A_bal_lr10x`
5. `S4_baseline_lr1x_full`, `S4_A_bal_lr10x`

Each candidate gets one fresh worker. The worker measures rank `0` cold,
then rank `0` again and ranks `1-9` warm, for 11 primary measurements.

## GPU monitoring and retries

The worker polls physical GPU 3 approximately once per second and records
compute-process snapshots under the review artifact root. The worker process
tree is allow-listed. An unrelated GPU process contaminates the active timing
interval.

- Contaminated warm measurements are retained as retry attempts and rerun after
  GPU 3 is clear in the same persistent worker.
- A contaminated cold measurement causes the current worker to stop; the
  candidate/configuration sequence is relaunched in a fresh worker so the cold
  CUDA context remains valid.
- Only clean primary attempts are selected by `primary_attempts.json` for
  aggregation and Dice evaluation. All contaminated attempts and monitor logs
  remain available for audit.

## Timing and evaluation

Timing is `case_load_complete` through `dvf_file_ready`, including optimization
and dense-DVF preparation/writing and excluding metric evaluation. The report
records cold time, warm mean and sample SD, cold-minus-warm gap, setup-device
time, expansion/backward gaps, actual steps, peak memory, backend metadata,
monitor status, and Git provenance.

Dice evaluation is separate and sequential on GPU 3 using the saved DVFs. It
evaluates one unique output for each of 10 candidates and 10 cases, for 100
rows. Evaluation time is excluded from timing.

## Expected acceptance

- `110/110` clean primary timing rows: 10 candidates x 11 measurements.
- `100/100` Dice rows: 10 candidates x 10 unique cases.
- Requested/effective backend flags match in every primary timing JSON.
- Every accepted DVF is finite `float32`, normalized-grid XYZ, shape
  `(160,224,192,3)`.
- The report explicitly labels the result first-10 diagnostic and does not
  claim OASIS-100 accuracy or authorize camera-ready edits.

## Artifact locations

- Host run root:
  `/mnt/shengdata1/hengjie/workspace/inrorspline/runs/oasis_table1_base_fast_ft_first10_gpu3_20260916_v1`
- Host review root:
  `/mnt/shengdata1/hengjie/workspace/inrorspline/repo_experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3`
- Container run root:
  `/database/inrorspline/runs/oasis_table1_base_fast_ft_first10_gpu3_20260916_v1`
- Container review root:
  `/database/inrorspline/repo_experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3`

## Commands

The full command is run inside the default container from
`/homebase/code_sync/inrorspline` as UID/GID `1009:1009`:

```bash
python scripts/experiments/oasis_alpha_sweep/run_mrdbscp_cudnn_ablation.py \
  --gpu 3 \
  --container-name py310sitreg_v1_artifact_store \
  --oasis-root /mnt/shengdata1/hengjie/workspace/wavereg/datasets/OASIS/OASIS_official/oasis_organized/split_v1 \
  --case-ranks 0-9 \
  --candidates MR_baseline_lr1x_full,MR_A_fast_decay,D2_baseline_lr1x_full,D2_A_bal_lr10x,D4_baseline_lr1x_full,D4_A_bal_lr10x,S2_baseline_lr1x_full,S2_A_bal_lr10x,S4_baseline_lr1x_full,S4_A_bal_lr10x \
  --configs benchmark_false_deterministic_true \
  --repeat-cold-rank \
  --variant oasis_table1_base_fast_ft_first10_gpu3_20260916_v1 \
  --out-root /database/inrorspline/runs/oasis_table1_base_fast_ft_first10_gpu3_20260916_v1 \
  --review-root /database/inrorspline/repo_experiments/2026-09-16_oasis_table1_base_fast_ft_first10_gpu3
```
