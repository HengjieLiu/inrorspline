# LungCT Case01 pTVReg Fit-Modes LR Sweep Technical Log

## Known Limitations

- This is a single-case diagnostic, not an all-case LungCT result.
- The field-fit target is reconstructed on the repo's reg1mm grid from the
  saved pTVReg native/full displacement field. The original pTVReg internal
  reg1mm displacement is not available in the `.mat`.
- Image-registration objectives are not identical across method families:
  IDIR uses sampled NCC plus IDIR BE, while spline methods use pTV LCC plus
  isoTV.
- pTVReg itself uses additional implementation details not matched here,
  including pTVReg's optimizer, pyramid behavior, linear basis, and MATLAB
  interpolation routines.

## Initial Protocol

- Case: 4DCT case `01`, fixed `T00`, moving `T50`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case01_T50_to_T00_deformation.mat`.
- Objective count: `24` jobs = `2 objectives x 4 methods x 3 learning rates`.
- Initial method/GPU map: IDIR `0`, SINR cps4 `1`, DirectCP cps4 `2`,
  DirectCP 4-stage `3`.
- Field-fit MSE domain: fixed T00 lung mask in the 1mm crop.
- Registration weights: IDIR `alpha_bending=10`, pTV isoTV `0.11`.

## Validation Notes

- pTVReg target validation must reproduce source `TRE_phys=0.7671410824`.
- Aggregation should report pTVReg-comparable original/native-crop snap TRE
  plus cropped-1mm diagnostic TRE for every completed row.

## 2026-06-26 Smoke Validation

- `python -m py_compile` passed for the new runner, launcher, and aggregator.
- Target-only validation reproduced source pTVReg `TRE_phys=0.7671410824122021`
  versus saved `0.7671410824122017`.
- Launcher dry-run produced exactly `24` jobs with fixed method/GPU assignment:
  IDIR `0`, SINR cps4 `1`, DirectCP cps4 `2`, DirectCP 4-stage `3`.
- Smoke output root:
  `/database/inrorspline/runs/lungct_case01_ptvreg_fit_modes_lr_sweep_smoke_v1`.
- Smoke review root:
  `/database/inrorspline/runs/lungct_case01_ptvreg_fit_modes_lr_sweep_smoke_review`.
- Smoke command used `--epochs 2`, `--stage-epochs 1,1,1,1`, and
  `--eval-interval 1`.
- Smoke aggregation completed `24/24` rows with validation `PASS`.

## 2026-06-26 Restart Scheduling Change

- The initial full launch used method-pinned scheduling. It was stopped because
  the methods have imbalanced runtimes.
- Orphaned child jobs from the stopped launch were terminated inside the Docker
  container, and all GPUs were confirmed idle before restart.
- The launcher default was changed to `method_batches`: run one method at a
  time, distributing that method's six objective/LR jobs across GPUs `0,1,2,3`.
- Dry-run confirmed the new order and GPU distribution.
- Full restart should use `--overwrite` to replace partial outputs from the
  stopped method-pinned launch.

## 2026-06-26 Full Run Completion

- Full method-batch launch completed `24/24` rows with validation `PASS`.
- Output root:
  `/database/inrorspline/runs/lungct_case01_ptvreg_fit_modes_lr_sweep_v1`.
- Review artifacts:
  `summary_runs.csv`, `summary_runs.json`, `review_report.md`, and plots under
  `plots/`.
- Preliminary IDIR/SINR-only preview plots were written under
  `prelim_idir_sinr/plots/`.

## 2026-06-26 Best-LR Review Figures

- Added best-LR overlay figures:
  `plots/best_lr_masked_mse_curves.png`,
  `plots/best_lr_registration_loss_idir.png`, and
  `plots/best_lr_registration_loss_spline_methods.png`.
- Best LR is selected per objective/method using original/native-crop snap TRE
  for registration and masked field MSE in voxel units for pTVReg field fitting.
- DirectCP 4-stage registration histories are plotted with reconstructed global
  epoch from stage schedule `400,400,800,900`, giving a continuous 1-2500 epoch
  axis.
