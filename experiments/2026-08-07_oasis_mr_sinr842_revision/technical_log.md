# Technical Log: OASIS MR-SINR 8/4/2 Revision

## Known Limitations Before Launch

- This pilot uses one OASIS validation/test pair only: seed-3 rank `0`,
  `(0333, 0115)`. It is a technical stability and implementation-orientation
  check, not a population claim.
- `mr_sinr842_composed` trains analytic B-spline-regularized residual SINR
  stages, but its final composed deformation is not itself a single B-spline
  field. Final-deformation BE/diffusion should therefore be interpreted as
  posthoc dense-field regularity, while per-stage losses record the analytic
  B-spline training regularizer.
- `mr_sinr842_requery` is a continued-generator ablation, not exact
  knot-insertion continuation. Stage-transition jumps are required diagnostics.
- Do not launch first10 or first100 scale-up until both pilot methods complete,
  the diagnostics are finite, and the composed method is inspected against the
  re-query ablation.

## 2026-08-07 Implementation Start

- Added runner support for `mr_sinr842_composed` and `mr_sinr842_requery`.
- Added `launch_oasis_mr_sinr842_pilot.py` for fixed rank-0 commands.
- Added `aggregate_oasis_mr_sinr842_pilot.py` for compact comparison,
  diagnostic collection, and a promotion decision record.

## 2026-08-07 Smoke And Launch

- Compile checks passed in Docker for the modified runner, launcher, and
  aggregator.
- Rank-0 case validation passed: pair `(0333, 0115)`, original sorted index
  `158`.
- Three-epoch smoke runs completed for both new methods under
  `<container-data-root>/inrorspline/runs/oasis_mr_sinr842_revision_smoke_v1`.
- Smoke aggregation passed with both methods technically complete and with the
  expected diagnostics present.
- Full 2500-epoch pilot launched detached through
  `launch_oasis_mr_sinr842_pilot.py --gpus 1,2 --parallel --launch --aggregate-after`.
  Initial logs reached stage 0 for both methods.
