# OASIS Rank-9 SITReg-Field Fit LR Sweep Execution Spec

## Status

Prepared for implementation and launch.

## Objective

Reuse the completed rank-9 image-registration LR sweep and add direct fitting
to the saved BS4 SITReg dense displacement field. The goal is to compare how
IDIR, SINR cps4, and DirectCP cps4 fit a SITReg registration field under the
same case, learning-rate, and mask protocol used for the Greedy-field fitting
diagnostic.

## Dataset

- Dataset: OASIS BrainMR `pair_test_200`.
- Pair order: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case rank: `9`.
- Original sorted index: `88`.
- Fixed/moving pair: `(0173, 0136)`.
- Brain-mask convention: fixed raw image-positive foreground,
  `fixed_raw > 0`, after standard OASIS reorientation.

## Matrix

- Reused objective:
  - `image_registration`: the 9 completed runs from
    `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`.
- New objective:
  - `sitreg_field_fit`: direct normalized-grid MSE to the saved BS4 SITReg
    dense displacement over the fixed-image-positive brain mask, with no image
    loss and no regularizer.
- Methods:
  - IDIR original dense SIREN, alpha `1`.
  - SINR cps4, alpha `1000`.
  - DirectCP cps4, alpha `1000`.
- Learning rates: `1e-2,1e-3,1e-4`.
- New field-fit epochs: `2500`.
- New field-fit eval interval: `250`.
- Seed: `1`.
- New jobs: `9`.
- Combined aggregation rows: `18`.

## Mask Definition

- Brain mask: raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`.
- SITReg-field fitting uses this mask for MSE. IDIR samples `10,000` masked
  coordinates per epoch; SINR/DirectCP compute MSE over all masked voxels after
  expanding cps4 controls.
- Reused image-registration IDIR used this same mask for sampled coordinates
  and sampled autograd BE points.
- Reused image-registration SINR/DirectCP keep the method-native full-image NCC
  objective; their regularity summaries use the fixed-image-positive
  foreground mask.
- Segmentation foreground, `seg > 0`, is only for label metrics
  Dice/HD95/ASD and visual/comparison audits.

## SITReg Target Convention

The field target is the saved BS4 SITReg best-Dice displacement:

```text
/database/wavereg/wipmidl_0407/OASIS_v1/round1_MIDL/gpu8_OASIS_v1_bs4_sitreg_v0_run1_20250407165940/inference/inference_bestdice/disp_0173_0136.nii.gz
```

Expected loader audit:

- raw saved shape: `(3,160,224,192)`
- storage layout: channel-first `zyx` repo/RAS
- component order: `zyx_voxel`
- reorientation: none
- source units: voxel displacement
- training target units: normalized-grid displacement in repo/PyTorch order

## Paths

- Large field-fit output root:
  `/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`
- Reused image-registration root:
  `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`
- Review root:
  `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep`
- Host large output equivalent:
  `/data/hengjie/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`

## Commands

Dry-run:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_sitreg_fit_modes_lr_sweep.py --dry-run --gpus 2,3 --max-concurrent 2
```

Smoke, using a separate output root:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_sitreg_fit_modes_lr_sweep.py \
  --gpus 2,3 \
  --max-concurrent 2 \
  --epochs 2 \
  --eval-interval 1 \
  --out-root /database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_smoke \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/smoke_review \
  --overwrite
```

Full launch:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_sitreg_fit_modes_lr_sweep.py \
  --gpus 2,3 \
  --max-concurrent 2 \
  --epochs 2500 \
  --eval-interval 250
```

Rerun aggregation only:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/aggregate_rank009_sitreg_fit_modes_lr_sweep.py
```

## Expected Review Outputs

- `summary_runs.csv/json`
- `loss_history_all.csv`
- `loss_history_augmented.csv`
- `final_metric_summary.csv`
- `image_registration_loss_curves.pdf`
- `sitreg_field_mse_curves.pdf`
- `final_metric_contact_sheet.pdf`
- `review_report.md`
