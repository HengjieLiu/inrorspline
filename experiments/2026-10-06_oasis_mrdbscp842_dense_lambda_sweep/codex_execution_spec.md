# OASIS MR-D-BSCP 8-4-2 to dense-field lambda sweep

## Objective

Evaluate direct full-resolution dense-field refinement after the unchanged
camera-ready MR-D-BSCP cps8/4/2 prefix on shuffled OASIS test rank 0. The
dense stage replaces cps1 B-spline controls and analytic B-spline bending
energy with a directly optimized voxel displacement and unmasked
forward-valid finite-difference bending energy.

## Frozen protocol

- Pair: fixed `0333`, moving `0115`, moving-to-fixed, shuffled rank `0`,
  original test index `158`.
- Prefix: cps `8,4,2`; downsample `4,2,1`; lambda `1000`; maximum epochs
  `400,800,1300`; LR `3e-2,1e-2,3e-3`.
- Dense stage: exact full-resolution expansion of the final cps2 displacement,
  followed by detach/clone into a `[160,224,192,3]` Adam parameter. LR is
  `1e-3`; maximum epochs are `1300`; lambda is swept over
  `10,100,1000,10000,100000`.
- Dense objective: full-image global NCC plus lambda times unmasked
  forward-valid finite-difference bending energy with component-mean
  reduction. No cps1 controls, knot insertion, or analytic B-spline
  regularizer are permitted in the final stage.
- Adaptive minimum epochs are `40,80,130,130`, patience is `20,35,50,50`,
  check interval is `25`, relative total threshold is `1e-4`, and relative
  data threshold is `5e-5`.
- Adam is reset at every stage. Seed is `1`; cuDNN benchmark is disabled and
  deterministic mode is enabled.
- Metrics use the canonical OASIS label-presence rule, 1-mm HD95, and
  area-weighted symmetric ASD.

## Execution

Run `launch_oasis_mrdbscp842_dense_lambda_sweep.py` inside
`py310sitreg_v1_artifact_store` as UID/GID `1009:1020`. Initially assign dense
lambda 10/100/10000/1000 to physical GPU 0/1/2/3, then send 100000 to the
first released GPU. Existing jobs are not modified.

Generated runs belong under
`${ARTIFACT_ROOT}/runs/oasis_seed3_rank000_mrdbscp842_dense_lambda_sweep_v1/`;
reports belong under the matching `${ARTIFACT_ROOT}/repo_experiments/` path.
The prior cps1 B-spline sweep is read-only comparison data.

## Acceptance

All five completion markers and finite `[160,224,192,3]` DVFs must exist.
Every summary must identify the final stage as `dense_voxel`, have null final
cps/B-spline metadata, contain an exact-zero cps2-to-dense initialization
audit, and match the frozen case, schedule, lambda, seed, and backend flags.
The first three stage stopping epochs and terminal losses must match across
candidates within `1e-6`. Aggregate only after all checks pass.
