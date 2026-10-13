# OASIS rank-0 VFA/SITReg B-spline approximation benchmark

Status: completed and technically accepted on 2026-10-06.

This experiment measures how accurately independent direct cubic B-spline
fields at control-point spacings `8`, `4`, `2`, and `1` can approximate the
saved best-Dice VFA and SITReg displacement fields for shuffled OASIS rank 0.
Generated fields, checkpoints, logs, figures, and reports remain external.

## Frozen protocol

- Case: fixed `OASIS_OAS1_0333`, moving `OASIS_OAS1_0115`, moving to fixed.
- Split: `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`, rank 0,
  original sorted index 158.
- Targets: canonical best-Dice VFA and SITReg saved fields, audited as
  channel-first `zyx` voxel displacements and converted to normalized-grid
  `xyz` for B-spline expansion.
- Candidate matrix: 2 targets x 4 cps values x learning rates
  `1e-2,3e-3,1e-3`, for 24 independent zero-initialized fits.
- Objective: mean squared residual over every component of every voxel after
  normalized-grid-to-voxel scaling; no mask, image loss, or regularizer.
- Optimizer: fresh Adam, seed 1, cuDNN benchmark off/deterministic on.
- Stopping: maximum 5000, minimum 500, check every 50, median window 50,
  patience 200, relative MSE plateau threshold `1e-6`.
- Selection: lowest full-volume voxel MSE, then normalized-grid MSE, then
  lower learning rate.
- Recovery: if selected nested-space error rises by more than 0.1% from a
  coarser to a finer cps, continue the offending best checkpoint for at most
  5000 more epochs at one-tenth its selected learning rate.
- Execution: longest fits first, one new worker per physical GPU across
  GPUs 0-3, without stopping or altering existing jobs.

## External outputs

- Run root:
  `${ARTIFACT_ROOT}/runs/oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1`
- Review root:
  `${ARTIFACT_ROOT}/repo_experiments/oasis_seed3_rank000_vfa_sitreg_bspline_cps8421_fullvoxel_mse_v1`

## Acceptance

The report is accepted only after 24 initial candidates and eight selected
fits pass field-shape, finite-value, source-hash, case/direction, full-volume
objective, no-regularizer, checkpoint, canonical metric, and nested-space
ordering checks. One-step GPU smoke runs cover all eight target/cps pairs
before production.

## Accepted result

All 24 candidates completed on their first attempts. The selected voxel MSEs
for VFA at cps `8/4/2/1` are `0.1146656573`, `0.0288353432`,
`0.0053879051`, and `1.6147702e-7`; the corresponding SITReg values are
`0.1258704364`, `0.0267495271`, `0.0037603469`, and `1.2670101e-7`.
Nested-space ordering passed without a recovery continuation. The external
review root contains all-candidate and selected CSV/JSON/Markdown reports plus
MSE/RMSE/rRMSE curves.
