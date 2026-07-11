---
doc_type: fit_modes_relative_field_fit_review
status: active
---

# Fit-Modes Relative Field-Fit Review

This note records the relative target-field fitting table used for the
preliminary fit-modes figure. Values come from:

- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_relative_field_fit_curve_summary.csv`
- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_target_power_summary.csv`

Definitions:

- `rMSE = mean((u_pred - u_target)^2) / (mean(u_target^2) + 1e-12)`
- `rRMSE = sqrt(rMSE)`, reported below as percent
- Means use the same masked normalized-grid displacement convention as the
  original field-fit objective.

## Target Powers

| Target | Target power |
|---|---:|
| OASIS rank-9 SITReg | 5.664249e-04 |
| OASIS rank-9 Greedy | 6.675104e-04 |
| LungCT case01 pTVReg | 7.440061e-04 |
| LungCT case08 pTVReg | 7.239231e-03 |

## Relative Error Table

| Target | Method | Start rMSE | Epoch 10 rMSE | End rMSE | Start rRMSE | Epoch 10 rRMSE | End rRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| OASIS rank-9 SITReg | IDIR | 7.884 | 0.616 | 2.439e-02 | 280.79% | 78.46% | 15.62% |
| OASIS rank-9 SITReg | SINR cps4 | 5.154 | 2.170 | 1.522e-02 | 227.03% | 147.31% | 12.34% |
| OASIS rank-9 SITReg | DirectCP cps4 | 1.000 | 0.135 | 1.373e-02 | 100.00% | 36.72% | 11.72% |
| OASIS rank-9 Greedy | IDIR | 6.262 | 0.625 | 9.011e-03 | 250.23% | 79.09% | 9.49% |
| OASIS rank-9 Greedy | SINR cps4 | 3.983 | 1.701 | 4.690e-03 | 199.57% | 130.44% | 6.85% |
| OASIS rank-9 Greedy | DirectCP cps4 | 1.000 | 0.120 | 3.126e-03 | 100.00% | 34.70% | 5.59% |
| LungCT case01 pTVReg | IDIR | 3.586 | 1.438 | 3.642e-03 | 189.36% | 119.94% | 6.03% |
| LungCT case01 pTVReg | SINR cps4 | 2.332 | 0.916 | 3.305e-04 | 152.71% | 95.72% | 1.82% |
| LungCT case01 pTVReg | DirectCP cps4 | 1.000 | 5.894e-02 | 3.826e-05 | 100.00% | 24.28% | 0.62% |
| LungCT case01 pTVReg | DirectCP 4stage cps4 | 1.000 | n/a | 3.825e-05 | 100.00% | n/a | 0.62% |
| LungCT case08 pTVReg | IDIR | 0.785 | 8.150e-02 | 3.238e-04 | 88.57% | 28.55% | 1.80% |
| LungCT case08 pTVReg | SINR cps4 | 0.840 | 0.159 | 1.595e-04 | 91.65% | 39.84% | 1.26% |
| LungCT case08 pTVReg | DirectCP cps4 | 1.000 | 0.267 | 1.349e-05 | 100.00% | 51.63% | 0.37% |
| LungCT case08 pTVReg | DirectCP 4stage cps4 | 1.000 | n/a | 1.338e-05 | 100.00% | n/a | 0.37% |

`n/a` means the finest-domain DirectCP 4-stage diagnostic was not logged at
epoch 10; coarse stages log that diagnostic at epoch 1 and then every 50 epochs.

## Review Notes

- The relative error normalization removes target-field amplitude scale, but it
  does not make all targets equally easy to represent or optimize.
- In these field-fit diagnostics, LungCT pTVReg target fields are fit to lower
  final relative error than the OASIS SITReg and Greedy targets. This is a
  field-fitting observation and should not be read as LungCT image registration
  being easier.
- The preliminary relative figure uses shared y-limits across target panels.
  The rRMSE row is capped at `10^2` percent for direct visual comparison.

## rRMSE Learning-Rate Trajectory Figure

Additional review artifacts compare requested learning-rate trajectories using
rRMSE percent only:

- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_rrmse_lr_trajectory_curves.png`
- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_rrmse_lr_trajectory_curves.pdf`
- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_rrmse_lr_trajectory_curve_summary.csv`
- `experiments/2026-06-26_fit_modes_paper_figure/fit_modes_rrmse_lr_trajectory_selected_runs.csv`

The figure compares IDIR and SINR cps4 at lr `1e-3`/`1e-4`,
DirectCP cps4 at lr `1e-2`/`1e-3`/`1e-4`, and LungCT DirectCP 4-stage at
lr `1e-2`/`1e-3`/`1e-4`. OASIS DirectCP 4-stage panels are explicitly marked
`not run`.

Validation summary:

- `40` requested LR/case/method rows are recorded; `34` are available curves
  and `6` are OASIS DirectCP 4-stage not-run entries.
- The four new LungCT DirectCP 4-stage diagnostic reruns each have `2500`
  history rows and `935` finite finest-domain diagnostic points.
- DirectCP cps4 and DirectCP 4-stage curves start at `100%` rRMSE, as expected
  from zero-control initialization.
- The rRMSE y-axis is shared across all panels and capped at `100%`.
