# OASIS Rank-9 Registration vs SITReg-Field Fit LR Sweep

- Generated: `2026-06-26 15:36:06 +0000`
- Field output root: `/database/inrorspline/runs/oasis_rank009_sitreg_fit_modes_lr_sweep_v1`
- Reused image-registration root: `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep`
- Completed summaries: `18` / `18`
- Case: rank `9`, original sorted index `88`, fixed/moving `(0173, 0136)`
- Objectives: reused image registration with BE regularization; fixed-image-positive brain-mask SITReg-field MSE with no regularizer.
- SITReg target: `bs4_sitreg_v0` best-Dice saved dense displacement, channel-first `zyx` repo/RAS, no reorientation.
- Brain mask: raw fixed OASIS image > 0 after standard reorientation; segmentations are used only for label metrics.
- IDIR display-scale regularization curves are diagnostic only; IDIR image-registration training preserves original BE scale.

## Acceptance

- PASS: all expected jobs are present with expected case, objective, metadata, and artifacts.

## Best Final Points

- Best image-registration Dice: SINR cps4 lr `0.0001` Dice `0.8296`.
- Best SITReg-field brain-mask MSE: DirectCP cps4 lr `0.01` MSE `7.77649e-06`.

## Files

- `summary_runs.csv/json`: one row per completed run.
- `loss_history_all.csv`: raw per-epoch history with run context.
- `loss_history_augmented.csv`: history with weighted regularization and IDIR display-scale diagnostics.
- `final_metric_summary.csv`: final metric table for quick plotting.
- `image_registration_loss_curves.pdf`: reused Exp1 loss curves.
- `sitreg_field_mse_curves.pdf`: SITReg-field MSE curves.
- `final_metric_contact_sheet.pdf`: final Dice/NCC/MSE panels by learning rate.

