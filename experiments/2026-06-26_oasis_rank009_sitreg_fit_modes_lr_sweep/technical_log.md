# OASIS Rank-9 SITReg-Field Fit LR Sweep Technical Log

## Status

- Planned and implemented on 2026-06-26.
- Implementation target: repo-local OASIS alpha-sweep utilities.
- Docker target: `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Requested launch GPUs: CUDA `2,3`, maximum `2` concurrent jobs.

## Known Limitations Before Launch

- This is a single-case diagnostic using OASIS seed-3 rank `9`
  `(0173,0136)`, selected to match the Greedy-field fitting diagnostic.
- Image-registration rows are reused from the completed Greedy experiment
  because that objective is independent of the saved dense-field target.
- SITReg-field fitting uses normalized-grid MSE over the fixed-image-positive
  brain mask. It does not use image loss, label metrics, or regularization
  during training.
- The brain mask is the raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`. Segmentation foreground, `seg > 0`, is reserved for
  Dice/HD95/ASD and visual audits.
- IDIR direct field fitting samples `10,000` fixed-image-positive coordinates
  per epoch. SINR/DirectCP expand their cps4 dense field each epoch and compute
  MSE on all fixed-image-positive voxels.
- IDIR BE scaling is not changed during image-registration training. Any IDIR
  scale correction is a plot-only diagnostic derived from the reused history.

## Acceptance Checks Planned

- Exactly `9` new SITReg field-fit summaries and `9` reused image-registration
  summaries for `18` combined rows.
- Every row resolves to case rank `9`, original sorted index `88`, fixed/moving
  `(0173,0136)`.
- SITReg target audit reports raw shape `(3,160,224,192)`, channel-first `zyx`
  repo/RAS layout, no reorientation, voxel source units, and normalized-grid
  training target units.
- Field-fit rows use MSE-only training on the fixed-image-positive brain mask.
  Full-volume MSE is retained as a posthoc context metric, not the training
  objective.
- Every new field-fit run has `DONE`, `summary.json`, `history.csv`,
  `case_metrics.csv`, `regularity_metrics.csv`, `be_metrics.csv`,
  `diffusion_metrics.csv`, `model_state.pt`, `mask_audit.json`,
  `target_audit.json`, and final dense displacement output.

## Launch Log

- `python -m py_compile` passed inside Docker for the new/modified experiment
  scripts.
- Target validation passed for rank `9`: original sorted index `88`,
  fixed/moving `(0173,0136)`.
- SITReg target audit matched the requested convention: raw shape
  `3,160,224,192`, channel-first `zyx` repo/RAS layout, no reorientation,
  voxel source units, and normalized-grid target shape `160,224,192,3`.
- Dry-run produced exactly `9` SITReg field-fit commands on GPUs `2,3` with
  `--max-concurrent 2`.
- Smoke launch completed `9/9` SITReg field-fit jobs for `2` epochs and
  combined aggregation passed with `18` rows and `errors=[]`.
- Full launch started at `2026-06-26 15:16:31 +0000`, completed aggregation at
  `2026-06-26 15:24:51 +0000`, and passed with `18/18` combined summaries,
  `9/9` new field-fit `DONE` files, `45,000` history rows, and `errors=[]`.

## Results Snapshot

- Best reused image-registration Dice: SINR cps4, lr `1e-4`, Dice `0.8296`.
- Best SITReg-field brain-mask MSE: DirectCP cps4, lr `1e-2`, MSE
  `7.77649e-06`.
- Review plots written:
  - `image_registration_loss_curves.pdf`
  - `sitreg_field_mse_curves.pdf`
  - `final_metric_contact_sheet.pdf`
