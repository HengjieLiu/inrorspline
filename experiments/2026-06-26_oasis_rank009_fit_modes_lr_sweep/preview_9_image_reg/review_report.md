# OASIS Rank-9 Registration vs Greedy-Field Fit LR Sweep

- Generated: `2026-06-26 05:22:47 +0000`
- Output root: `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/preview_9_image_reg`
- Completed summaries: `9` / `18`
- Case: rank `9`, original sorted index `88`, fixed/moving `(0173, 0136)`
- Objectives: image registration with BE regularization; full-volume Greedy-field MSE with no regularizer.
- IDIR display-scale regularization curves are diagnostic only; IDIR training preserves original BE scale.

## Acceptance

- FAIL: Missing run combo: objective=greedy_field_fit, method=direct_cp_cps4_field_mse, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=direct_cp_cps4_field_mse, alpha=1000, lr=0.001
- FAIL: Missing run combo: objective=greedy_field_fit, method=direct_cp_cps4_field_mse, alpha=1000, lr=0.01
- FAIL: Missing run combo: objective=greedy_field_fit, method=idir_original_dense_siren_field_mse, alpha=1, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=idir_original_dense_siren_field_mse, alpha=1, lr=0.001
- FAIL: Missing run combo: objective=greedy_field_fit, method=idir_original_dense_siren_field_mse, alpha=1, lr=0.01
- FAIL: Missing run combo: objective=greedy_field_fit, method=sinr_cps4_field_mse, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=sinr_cps4_field_mse, alpha=1000, lr=0.001
- FAIL: Missing run combo: objective=greedy_field_fit, method=sinr_cps4_field_mse, alpha=1000, lr=0.01

## Best Final Points

- Best image-registration Dice: SINR cps4 lr `0.0001` Dice `0.8296`.

## Files

- `summary_runs.csv/json`: one row per completed run.
- `loss_history_all.csv`: raw per-epoch history with run context.
- `loss_history_augmented.csv`: history with weighted regularization and IDIR display-scale diagnostics.
- `final_metric_summary.csv`: final metric table for quick plotting.
- `image_registration_loss_curves.pdf`: Exp1 loss curves.
- `greedy_field_mse_curves.pdf`: Exp2 MSE curves.
- `final_metric_contact_sheet.pdf`: final Dice/NCC/MSE panels by learning rate.

