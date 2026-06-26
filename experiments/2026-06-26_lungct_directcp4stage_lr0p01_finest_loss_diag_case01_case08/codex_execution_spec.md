# LungCT DirectCP 4-Stage Finest-Domain Loss Diagnostic

## Objective

Rerun DirectCP 4-stage image registration for 4DCT cases `01` and `08` at
`lr=0.01` while preserving the original stage-wise training objective. Record an
additional diagnostic loss on the full 1mm crop at every epoch.

## Implementation

- Add `--finest-loss-diagnostic` and `--finest-loss-every` to
  `scripts/experiments/lungct_no_sweep_repro/run_one.py`.
- Diagnostic loss is pre-optimizer-step, wrapped in `torch.no_grad()`, and does
  not affect backprop.
- Coarse-stage diagnostics expand the current controls to the full 1mm crop via
  the existing cubic B-spline analytic expansion path.
- Add
  `scripts/experiments/lungct_no_sweep_repro/plot_finest_domain_loss_diagnostic.py`
  to write companion plots next to the accepted Case01/Case08 sweep plots.

## Launch Defaults

- Output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_v1`.
- Smoke output root:
  `/database/inrorspline/runs/lungct_directcp4stage_lr0p01_finest_loss_diag_case01_case08_smoke_v1`.
- Cases: `01`, `08`.
- GPUs: case01 on GPU `0`, case08 on GPU `1`.
- Full stage epochs: `400,400,800,900`.
- Smoke stage epochs: `1,1,1,1`.

## Acceptance

- Compile checks pass for the modified runner and plotter.
- Smoke histories contain finite `finest_loss` values and
  `finest_loss_domain=full_1mm_crop`.
- Full histories contain `2500` rows per case with finite diagnostic loss
  columns.
- New companion PNGs render next to the original accepted plots without
  overwriting them.
