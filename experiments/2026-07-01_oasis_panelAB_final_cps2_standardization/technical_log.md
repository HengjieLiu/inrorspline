# Technical Log

## Known Limitations

- This add-on is a single OASIS rank-9 diagnostic for the fitting figure, not a
  population-level OASIS rerun.
- It changes only OASIS single-stage spline final spacing for Panel A/B
  comparison. LungCT remains unchanged because it is already final-spacing
  matched at cps4.
- IDIR is not rerun; the final-control-spacing issue applies to spline-based
  `INR-BSCP`, `D-BSCP`, and `MR-D-BSCP`.
- OASIS Panel A cps2 image-registration histories are reused from compatible
  existing rank-9 runs rather than duplicated into a new Panel A run root.

## 2026-07-01 Plan

- Add cps2 support to OASIS rank-9 Greedy/SITReg field-fit one-job runners.
- Update paper artifact standards so `fitting_figure` writes to
  `v002_candidate`.
- Run smoke jobs for SITReg/Greedy x SINR cps2/DirectCP cps2.
- Run full four-job add-on using all visible GPUs.
- Regenerate v002 candidate fitting figures and comparison reports.

## 2026-07-01 Smoke

- Smoke root:
  `${DATA_ROOT}/runs/oasis_rank009_final_cps2_field_fit_smoke_v1`.
- Smoke jobs: `4/4` completed.
- Smoke epochs: `2`.
- Smoke verification: all four histories had two finite
  `field_mse_objective_normalized_grid` values.

## 2026-07-01 Full Run

- Full root:
  `${DATA_ROOT}/runs/oasis_rank009_final_cps2_field_fit_v1`.
- GPU policy: used all four visible GPUs, one job per GPU.
- Monitor cadence: `5` minutes.
- Full jobs: `4/4` completed with no failures.
- Full epochs: `2500` for each job.
- Field-fit history verification: all final values were finite.

Final masked target-field MSE values:

| Target | Method | Final MSE |
|---|---|---:|
| Greedy | `direct_cp_cps2_be_bspline_analytic` | `1.0078334611307582e-07` |
| SITReg | `direct_cp_cps2_be_bspline_analytic` | `1.133801561081782e-06` |
| Greedy | `sinr_cps2_be_bspline_analytic` | `1.9186304598406423e-06` |
| SITReg | `sinr_cps2_be_bspline_analytic` | `7.383104730251944e-06` |

## 2026-07-01 v002 Candidate Artifacts

- Generated fitting artifacts under
  `experiments/paper_artifacts/fitting_figure/v002_candidate/`.
- Executed the paper-layout Panel A/B notebook under
  `experiments/paper_artifacts/fitting_figure/v002_candidate/panelAB_composite/`.
- Generated comparison artifacts under
  `experiments/paper_artifacts/comparison/v001_vs_v002_candidate_panelAB_final_cps2/`.
