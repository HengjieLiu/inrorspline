# OASIS Rank-9 Fit-Mode LR Sweep Execution Spec

## Status

Prepared for implementation and launch.

## Objective

Compare three deformation parameterizations on one high-performing Greedy OASIS
case under two optimization targets: image registration and direct dense-field
fitting. The goal is a paper-facing diagnostic figure showing whether each
method fits an image-registration objective differently from a known strong
registration field.

## Dataset

- Dataset: OASIS BrainMR `pair_test_200`.
- Pair order: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case rank: `9`.
- Original sorted index: `88`.
- Fixed/moving pair: `(0173, 0136)`.
- Foreground convention for image-domain metrics: fixed raw image-positive
  foreground, `fixed_image > 0`.

## Matrix

- Methods:
  - IDIR original dense SIREN, alpha `1`.
  - SINR cps4, alpha `1000`.
  - DirectCP cps4, alpha `1000`.
- Learning rates: `1e-2,1e-3,1e-4`.
- Objectives:
  - `image_registration`: method-native image registration with BE
    regularization.
  - `greedy_field_fit`: direct normalized-grid MSE to canonical Greedy dense
    displacement over the fixed-image-positive brain mask, with no regularizer.
- Epochs: `2500`.
- Eval interval: `250`.
- Seed: `1`.
- Expected jobs: `18`.

## Mask Definition

- Brain mask: raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`.
- Image-registration IDIR uses this mask for its sampled coordinate domain and
  sampled autograd BE points.
- Image-registration SINR/DirectCP keep the method-native full-image NCC
  objective; their regularity summaries still use the fixed-image-positive
  foreground mask.
- Greedy-field fitting uses this same brain mask for MSE. IDIR samples `10,000`
  masked coordinates per epoch; SINR/DirectCP compute MSE over all masked voxels
  after expanding cps4 controls.
- Segmentation foreground, `seg > 0`, is only for label metrics
  Dice/HD95/ASD and visual/comparison audits.

## Greedy Target Convention

The Greedy field is loaded from the canonical saved result
`disp_0173_0136.nii.gz`. It must use the same convention as the saved-baseline
evaluation:

- raw storage: 5D channel-last NIfTI
- component reorientation: scalar OASIS LIA-to-RAS transform per component
- component signs: `-,-,+`
- training target: normalized-grid displacement in repo/PyTorch order

## Paths

- Large output root:
  `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`
- Review root:
  `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep`
- Host large output equivalent:
  `/data/hengjie/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`

## Commands

Dry-run:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_fit_modes_lr_sweep.py --dry-run --gpus 0,1,2
```

Smoke, using a separate output root:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_fit_modes_lr_sweep.py \
  --gpus 0,1,2 \
  --epochs 2 \
  --eval-interval 1 \
  --out-root /database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_smoke \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/smoke_review \
  --overwrite
```

Full launch:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/launch_rank009_fit_modes_lr_sweep.py \
  --gpus 0,1,2 \
  --max-concurrent 3 \
  --epochs 2500 \
  --eval-interval 250
```

Rerun aggregation only:

```bash
cd /homebase/code_sync/inrorspline
python scripts/experiments/oasis_alpha_sweep/aggregate_rank009_fit_modes_lr_sweep.py
```

## Expected Review Outputs

- `summary_runs.csv/json`
- `loss_history_all.csv`
- `loss_history_augmented.csv`
- `final_metric_summary.csv`
- `image_registration_loss_curves.pdf`
- `greedy_field_mse_curves.pdf`
- `final_metric_contact_sheet.pdf`
- `review_report.md`
