# LungCT Case08 pTVReg Fit-Modes LR Sweep Technical Log

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

- Case: 4DCT case `08`, fixed `T00`, moving `T50`.
- pTVReg source field:
  `/database/inrorspline/ptvreg/4DCT/case08_T50_to_T00_deformation.mat`.
- Objective count: `24` jobs = `2 objectives x 4 methods x 3 learning rates`.
- Scheduling: method batches; each method's six jobs are distributed across
  GPUs `0,1,2,3`.
- Field-fit MSE domain: fixed T00 lung mask in the 1mm crop.
- Registration weights: IDIR `alpha_bending=10`, pTV isoTV `0.11`.
- DirectCP 4-stage schedule: `32,16,8,4`, stage epochs `400,400,800,900`.

## Validation Notes

- pTVReg target validation must reproduce source
  `TRE_phys=1.014219831733267`.
- The saved target has `TREstd_case=1.2922227355399813`.
- Aggregation should report pTVReg-comparable original/native-crop snap TRE
  plus cropped-1mm diagnostic TRE for every completed row.
- Final review plots must match the Case01 review set, including best-LR
  overlays and continuous DirectCP 4-stage epoch axes.

## 2026-06-26 Preflight

- `python -m py_compile` passed for the generalized field-fit runner, launcher,
  and aggregator.
- Target-only validation reproduced source pTVReg
  `TRE_phys=1.0142198317332671` and
  `TREstd_case=1.2922227355399807`.
- Case08 pTVReg target shape on the reg1mm crop is `(228,207,254,3)`.
- Fixed T00 lung-mask MSE domain contains `5,313,087` voxels
  (`44.3208436%` of the reg1mm crop).
- Launcher dry-run produced exactly `24` Case08 jobs with method-batch
  scheduling across GPUs `0,1,2,3`.

## 2026-06-26 Smoke Validation

- Smoke output root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_smoke_v1`.
- Smoke review root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_smoke_review`.
- Smoke command used `--epochs 2`, `--stage-epochs 1,1,1,1`, and
  `--eval-interval 1`.
- Smoke aggregation completed `24/24` rows with validation `PASS`.
- Smoke review wrote all required plots, including best-LR masked-MSE and
  registration-loss figures.

## 2026-06-26 Full Run

- Full output root:
  `/database/inrorspline/runs/lungct_case08_ptvreg_fit_modes_lr_sweep_v1`.
- Review root:
  `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`.
- Full launch used method-batch scheduling across GPUs `0,1,2,3`, with
  `2500` epochs and DirectCP 4-stage epochs `400,400,800,900`.
- Aggregation completed `24/24` rows with validation `PASS`.
- Final review wrote:
  `summary_runs.csv`, `summary_runs.json`, `review_report.md`,
  `image_registration_loss_curves.png`, `ptvreg_field_mse_curves.png`,
  `final_metric_vs_lr.png`, `best_lr_masked_mse_curves.png`,
  `best_lr_registration_loss_idir.png`, and
  `best_lr_registration_loss_spline_methods.png`.
- Best pTVReg-field-fit masked MSE rows by method:
  IDIR `lr=1e-4`, SINR cps4 `lr=1e-4`, DirectCP cps4 `lr=1e-2`,
  DirectCP 4-stage cps4 `lr=1e-2`.
- Best image-registration original-snap TRE rows by method:
  IDIR `lr=1e-4`, SINR cps4 `lr=1e-3`, DirectCP cps4 `lr=1e-2`,
  DirectCP 4-stage cps4 `lr=1e-2`.
