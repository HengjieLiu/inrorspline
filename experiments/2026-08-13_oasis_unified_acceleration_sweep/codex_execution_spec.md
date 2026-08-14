# OASIS Unified Acceleration Sweep Execution Spec

## Purpose

Run a unified runtime-reduction screen for OASIS seed-3 first10 before selecting accelerated variants for first100 evaluation.

Primary first100 selection targets:

- MR-D-BSCP `8-4-2`
- D-BSCP `cps2`
- SINR `cps2`

Contextual follow-up targets after the primary phase:

- D-BSCP `cps4`
- SINR `cps4`

No first100 run is launched in this experiment. The output of this experiment is a first10 accuracy/runtime screen plus cold/warm timing summaries to guide model selection.

## Dataset And Protocol

- Dataset: OASIS BrainMR.
- Split: `pair_test_200`.
- Subset: first10 after seed-3 shuffle.
- Pair file: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case ranks: `0-9`.
- Foreground convention: raw fixed image-positive foreground, `fixed_image > 0`, for foreground regularity metrics.
- Segmentation masks are used for Dice, HD95, and ASD only.
- Alpha: `1000`.
- Seed: `1`.
- Docker/container: `py310sitreg_v1_test1`.

## Roots

- Output root: `${DATA_ROOT}/runs/oasis_unified_acceleration_sweep_alpha1000_v1`
- Review root: `experiments/2026-08-13_oasis_unified_acceleration_sweep/`
- Launcher: `scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py`
- Aggregator: `scripts/experiments/oasis_alpha_sweep/aggregate_oasis_unified_acceleration_sweep.py`

## Execution Model

Use all four GPUs, but do not parallelize the 10 cases within a candidate.

Each candidate runs in one persistent Python worker process. Inside that worker, ranks `0-9` run sequentially on one assigned GPU. Up to four candidate workers run concurrently, one per GPU.

This records:

- Cold runtime: rank `0`, the first case in a candidate worker.
- Warm runtime: mean/std over ranks `1-9`, after Python, CUDA, cuDNN, and PyTorch shape-specific paths have already been initialized in that worker.
- Amortized runtime: mean over ranks `0-9`.

The same cold/warm split is reported for:

- full registration time, including optimization, DVF save, evaluation, regularity metrics, and output packaging;
- loaded-images-to-DVF-file time, stopping immediately after the dense DVF has been written to disk.

## Unified Candidate Template

All methods use the same acceleration template, expressed relative to their method-native baseline learning rate.

Fixed schedules:

- `baseline_lr1x_full`: full 2500-epoch baseline.
- `F60_lr3x`: 60% epoch budget with 3x LR.
- `F40_lr10x`: 40% epoch budget with 10x LR.
- `F25_lr30x`: 25% epoch budget with 30x LR.

Adaptive schedules:

- `A_bal_lr10x`: full max budget, 10x LR, minimum 15% of baseline epochs, patience 6% of baseline epochs, `min_delta=1e-4`.
- `A_fast_lr30x`: full max budget, 30x LR, minimum 10% of baseline epochs, patience 4% of baseline epochs, `min_delta=1e-4`.

## Method-Specific Learning Rates

D-BSCP baseline LR is `1e-3`.

- `baseline_lr1x_full`: `1e-3`, 2500 epochs.
- `F60_lr3x`: `3e-3`, 1500 epochs.
- `F40_lr10x`: `1e-2`, 1000 epochs.
- `F25_lr30x`: `3e-2`, 600 epochs.
- `A_bal_lr10x`: `1e-2`, max 2500, min 375, patience 150.
- `A_fast_lr30x`: `3e-2`, max 2500, min 250, patience 100.

SINR baseline LR is `1e-4`.

- `baseline_lr1x_full`: `1e-4`, 2500 epochs.
- `F60_lr3x`: `3e-4`, 1500 epochs.
- `F40_lr10x`: `1e-3`, 1000 epochs.
- `F25_lr30x`: `3e-3`, 600 epochs.
- `A_bal_lr10x`: `1e-3`, max 2500, min 375, patience 150.
- `A_fast_lr30x`: `3e-3`, max 2500, min 250, patience 100.

MR-D-BSCP `8-4-2` baseline LR is `1e-3` and the baseline stage epoch schedule is `400,800,1300`.

Strict unified same-LR schedules:

- `MR_baseline_lr1x_full`: stage LR `1e-3,1e-3,1e-3`, stage epochs `400,800,1300`.
- `MR_F60_lr3x`: stage LR `3e-3,3e-3,3e-3`, stage epochs `240,480,780`.
- `MR_F40_lr10x`: stage LR `1e-2,1e-2,1e-2`, stage epochs `160,320,520`.
- `MR_F25_lr30x`: stage LR `3e-2,3e-2,3e-2`, stage epochs `100,200,300`.
- `MR_A_bal_lr10x`: stage LR `1e-2,1e-2,1e-2`, max `400,800,1300`, min `60,120,195`, patience `25,50,80`.
- `MR_A_fast_lr30x`: stage LR `3e-2,3e-2,3e-2`, max `400,800,1300`, min `40,80,130`, patience `20,35,50`.

MR-D-BSCP also includes decayed-stage LR variants because the previous best practical candidate used this schedule:

- `MR_F60_decay`: stage LR `3e-2,1e-2,3e-3`, stage epochs `240,480,780`.
- `MR_F40_decay`: stage LR `3e-2,1e-2,3e-3`, stage epochs `160,320,520`.
- `MR_F25_decay`: stage LR `3e-2,1e-2,3e-3`, stage epochs `100,200,300`.
- `MR_A_bal_decay`: stage LR `3e-2,1e-2,3e-3`, max `400,800,1300`, min `60,120,195`, patience `25,50,80`.
- `MR_A_fast_decay`: stage LR `3e-2,1e-2,3e-3`, max `400,800,1300`, min `40,80,130`, patience `20,35,50`.
- `MR_A_fast_decay_legacy`: previous best setting, stage LR `3e-2,1e-2,3e-3`, max `400,800,1300`, min `75,150,250`, patience `50,75,100`.

## Candidate Counts

Primary phase:

- MR-D-BSCP `8-4-2`: 12 candidates.
- D-BSCP `cps2`: 6 candidates.
- SINR `cps2`: 6 candidates.
- Total primary candidate workers: 24.
- Total primary registrations: 240.

cps4 context phase:

- D-BSCP `cps4`: 6 candidates.
- SINR `cps4`: 6 candidates.
- Total cps4 candidate workers: 12.
- Total cps4 registrations: 120.

## Selection Readout

Dice degradation is graded versus the same-method baseline:

- `no_loss_or_gain`: Dice is equal to or better than baseline.
- `small_0_to_0p02`: Dice drop is greater than 0 and no more than 0.02.
- `moderate_0p02_to_0p05`: Dice drop is greater than 0.02 and no more than 0.05.
- `large_gt_0p05`: Dice drop is greater than 0.05.

HD95 and ASD degradation are reported continuously. They are not used as artificial hard cutoffs in the primary screen, but candidates with material boundary degradation should be deprioritized.

Runtime readouts:

- full-registration cold rank0 time;
- full-registration warm ranks1-9 mean/std;
- loaded-images-to-DVF-file cold rank0 time;
- loaded-images-to-DVF-file warm ranks1-9 mean/std;
- peak GPU memory mean/max.

## Commands

Syntax check:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/run_one.py \
  ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/aggregate_oasis_unified_acceleration_sweep.py
```

Dry-run primary manifest:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  --phase primary --gpus 0,1,2,3 --max-concurrent 4 --dry-run
```

Dry-run cps4 manifest:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  --phase cps4 --gpus 0,1,2,3 --max-concurrent 4 --dry-run
```

Smoke all candidates with one epoch:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  --phase smoke --smoke-target-phase all --case-ranks 0 --epoch-override 1 \
  --gpus 0,1,2,3 --max-concurrent 4 --skip-aggregate --overwrite
```

Primary full screening:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  --phase primary --gpus 0,1,2,3 --max-concurrent 4 --allow-incomplete-aggregate
```

cps4 context screening after primary phase:

```bash
docker exec py310sitreg_v1_test1 python ${REPO_ROOT}/scripts/experiments/oasis_alpha_sweep/launch_oasis_unified_acceleration_sweep.py \
  --phase cps4 --gpus 0,1,2,3 --max-concurrent 4 --allow-incomplete-aggregate
```

## Acceptance Checks

- `py_compile` passes.
- Dry-run primary reports 24 candidates and 240 registrations.
- Dry-run cps4 reports 12 candidates and 120 registrations.
- Smoke runs complete and write `DONE`, `summary.json`, and `timing.json`.
- Full primary aggregation reports all expected summaries or explicitly lists missing/failed jobs.
- Full cps4 aggregation reports all expected summaries or explicitly lists missing/failed jobs.
- Reports include Dice degradation grades, HD95/ASD degradation, full registration cold/warm time, loaded-images-to-DVF-file cold/warm time, and peak GPU memory.
