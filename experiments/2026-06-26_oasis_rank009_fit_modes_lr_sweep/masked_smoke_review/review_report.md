# OASIS Rank-9 Registration vs Greedy-Field Fit LR Sweep

- Generated: `2026-06-26 05:46:44 +0000`
- Output root: `/database/inrorspline/runs/oasis_rank009_fit_modes_lr_sweep_masked_smoke`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_rank009_fit_modes_lr_sweep/masked_smoke_review`
- Completed summaries: `3` / `18`
- Case: rank `9`, original sorted index `88`, fixed/moving `(0173, 0136)`
- Objectives: image registration with BE regularization; fixed-image-positive brain-mask Greedy-field MSE with no regularizer.
- Brain mask: raw fixed OASIS image > 0 after standard reorientation; segmentations are used only for label metrics.
- IDIR display-scale regularization curves are diagnostic only; IDIR training preserves original BE scale.

## Acceptance

- FAIL: Missing run combo: objective=greedy_field_fit, method=direct_cp_cps4_field_mse, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=direct_cp_cps4_field_mse, alpha=1000, lr=0.01
- FAIL: Missing run combo: objective=greedy_field_fit, method=idir_original_dense_siren_field_mse, alpha=1, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=idir_original_dense_siren_field_mse, alpha=1, lr=0.01
- FAIL: Missing run combo: objective=greedy_field_fit, method=sinr_cps4_field_mse, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=greedy_field_fit, method=sinr_cps4_field_mse, alpha=1000, lr=0.01
- FAIL: Missing run combo: objective=image_registration, method=direct_cp_cps4_be_bspline_analytic, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=image_registration, method=direct_cp_cps4_be_bspline_analytic, alpha=1000, lr=0.001
- FAIL: Missing run combo: objective=image_registration, method=direct_cp_cps4_be_bspline_analytic, alpha=1000, lr=0.01
- FAIL: Missing run combo: objective=image_registration, method=idir_original_dense_siren_be_autograd, alpha=1, lr=0.0001
- FAIL: Missing run combo: objective=image_registration, method=idir_original_dense_siren_be_autograd, alpha=1, lr=0.001
- FAIL: Missing run combo: objective=image_registration, method=idir_original_dense_siren_be_autograd, alpha=1, lr=0.01
- FAIL: Missing run combo: objective=image_registration, method=sinr_cps4_be_bspline_analytic, alpha=1000, lr=0.0001
- FAIL: Missing run combo: objective=image_registration, method=sinr_cps4_be_bspline_analytic, alpha=1000, lr=0.001
- FAIL: Missing run combo: objective=image_registration, method=sinr_cps4_be_bspline_analytic, alpha=1000, lr=0.01

## Best Final Points

- Best Greedy-field brain-mask MSE: DirectCP cps4 lr `0.001` MSE `0.000597393`.

## Files

- `summary_runs.csv/json`: one row per completed run.
- `loss_history_all.csv`: raw per-epoch history with run context.
- `loss_history_augmented.csv`: history with weighted regularization and IDIR display-scale diagnostics.
- `final_metric_summary.csv`: final metric table for quick plotting.
- `image_registration_loss_curves.pdf`: Exp1 loss curves.
- `greedy_field_mse_curves.pdf`: Exp2 MSE curves.
- `final_metric_contact_sheet.pdf`: final Dice/NCC/MSE panels by learning rate.

