# OASIS Panel A/B Final-CPS Matched Field-Fit Add-On

## Objective

Generate the missing OASIS rank-9 cps2 field-fit trajectories needed for the
v002 candidate merged Panel A/B fitting figure. The comparison changes OASIS
single-stage `INR-BSCP` and `D-BSCP` from cps4 to cps2 so their final control
spacing matches `MR-D-BSCP` ending at cps2 in the `8-4-2` schedule.

LungCT remains unchanged because its multistage DirectCP schedule is
`32-16-8-4`, ending at cps4, which already matches the LungCT single-stage
cps4 rows.

## Dataset And Case

- Dataset: OASIS BrainMR `pair_test_200`.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Case rank: `9`.
- Original sorted index: `88`.
- Fixed/moving IDs: `0173,0136`.
- Paper-facing subtitle: `OASIS M0136->F0173`.
- Field-fit mask: fixed raw image-positive foreground, `fixed_raw > 0`.
- Target units: normalized-grid displacement.

## New Jobs

| Target | Method | LR | Alpha recorded | Epochs | Objective |
|---|---|---:|---:|---:|---|
| SITReg | `sinr_cps2_be_bspline_analytic` | `1e-4` | `1000` | `2500` | masked target-field MSE |
| SITReg | `direct_cp_cps2_be_bspline_analytic` | `1e-3` | `1000` | `2500` | masked target-field MSE |
| Greedy | `sinr_cps2_be_bspline_analytic` | `1e-4` | `1000` | `2500` | masked target-field MSE |
| Greedy | `direct_cp_cps2_be_bspline_analytic` | `1e-3` | `1000` | `2500` | masked target-field MSE |

Common settings:

- Seed: `1`.
- Eval interval: `250`.
- No image loss during field fitting.
- No regularizer during field fitting.
- SINR hidden dim/layers/omega use existing field-fit runner defaults.
- DirectCP cps2 expands cubic B-spline controls over the fixed-image-positive
  mask for objective MSE.

## Reused Inputs

Panel A OASIS cps2 image-registration histories are reused because they already
match the requested case and operating settings:

- `SINR cps2`: alpha `1000`, lr `1e-4`, epochs `2500`, seed `1`.
- `DirectCP cps2`: alpha `1000`, lr `1e-3`, epochs `2500`, seed `1`.

Existing OASIS IDIR and DirectCP `8-4-2` Panel A/B rows remain unchanged.

## Output Locations

- Smoke root:
  `${DATA_ROOT}/runs/oasis_rank009_final_cps2_field_fit_smoke_v1`
- Full root:
  `${DATA_ROOT}/runs/oasis_rank009_final_cps2_field_fit_v1`
- Review root:
  `experiments/2026-07-01_oasis_panelAB_final_cps2_standardization/`
- Candidate artifact root:
  `experiments/paper_artifacts/fitting_figure/v002_candidate/`
- Comparison root:
  `experiments/paper_artifacts/comparison/v001_vs_v002_candidate_panelAB_final_cps2/`

## Launch Policy

- Use the default Docker container: `py310sitreg_v1_test1`.
- Use all visible GPUs, one job per GPU.
- Run a 2-epoch smoke first.
- Poll full-run progress every 5 minutes and write monitor snapshots to the
  review root.
