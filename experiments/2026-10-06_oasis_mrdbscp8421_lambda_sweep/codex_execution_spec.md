# OASIS MR-D-BSCP 8-4-2-1 cps1 lambda sweep

## Objective

Evaluate a new cps1 refinement stage on shuffled OASIS test rank 0 while
keeping the camera-ready MR-D-BSCP 8-4-2 prefix unchanged. Sweep only the cps1
regularization weight and compare Dice, HD95, and symmetric ASD with the
previously verified MR-D-BSCP 8-4-2, Greedy, VFA, and SITReg results.

## Frozen protocol

- Pair: fixed `0333`, moving `0115`, moving-to-fixed, shuffled rank `0`,
  original test index `158`.
- Schedule: cps `8,4,2,1`; downsample `4,2,1,1`; maximum epochs
  `400,800,1300,1300`; LR `3e-2,1e-2,3e-3,1e-3`.
- Adaptive minimum epochs `40,80,130,130`, patience `20,35,50,50`, check
  interval `25`, relative total threshold `1e-4`, and relative data threshold
  `5e-5`.
- Stage lambda: `1000,1000,1000,X`, where cps1 `X` is one of
  `10,100,1000,10000,100000`.
- Adam is reset at every stage. Seed is `1`; cuDNN benchmark is disabled and
  deterministic mode is enabled.
- Metrics use the canonical OASIS label-presence rule, 1-mm HD95, and
  area-weighted symmetric ASD.

## Execution

Run `launch_oasis_mrdbscp8421_lambda_sweep.py` inside
`py310sitreg_v1_artifact_store` as UID/GID `1009:1009`. Add one worker to each
physical GPU. Initially assign cps1 lambda 10/100/10000/1000 to GPU 0/1/2/3,
then send 100000 to the first released GPU. Existing jobs are not modified.

Generated runs belong under
`${ARTIFACT_ROOT}/runs/oasis_seed3_rank000_mrdbscp8421_cps1_lambda_sweep_v1/`;
reports belong under the matching `${ARTIFACT_ROOT}/repo_experiments/` path.

## Acceptance

All five completion markers and finite `[160,224,192,3]` DVFs must exist.
Every summary must match the frozen case, schedule, stage lambda, seed, and
backend flags. The first three stage stopping epochs and terminal losses must
match across candidates within `1e-6`. Aggregate only after all checks pass.
