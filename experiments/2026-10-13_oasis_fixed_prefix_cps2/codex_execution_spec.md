# Fixed cps8/cps4 lambda1 to cps2 lambda sweep

## Approved scope

Run 60 fresh cps2 fits, seed3 shuffled first ten OASIS pairs, lambda
0.1/0.2/0.5/1/2/5. Keep each case's exact saved P43/P48 cps8/cps4 stages at
lambda1. Branch from stage1/best.pt (cps4), using historical dyadic knot
insertion to initialize cps2. Never initialize from an already optimized cps2
field. No cps1 or dense refinement, no historical-output changes.

## Frozen settings

Historical native raw intensities and pyramid factors4/2/1; local squared
NCC9 (1-mean correlation squared) plus unmasked total expanded-field voxel
forward-difference diffusion. Cps8/cps4 LR0.03/0.01, lambda1; reuse their
selected controls and records. Cps2 LR0.003 in normalized XYZ units, full
resolution, max/min1300/130, patience50, median/check25, relative total/data
thresholds1e-4/5e-5, no LR halving. Fresh Adam each cps2 fit; lowest-total-loss
checkpoint including initialization. Seed1, FP32, cuDNN F/T, TF32off.

## Execution and artifacts

Separate runner: scripts/experiments/oasis_alpha_sweep/fixed_prefix_cps2.py.
Artifact key: oasis_seed3_first10_mrdbscp84_prefixlambda1_cps2_lambda_sweep_v1.
Run/review roots: ${ARTIFACT_ROOT}/runs and ${ARTIFACT_ROOT}/repo_experiments.
One new worker each on GPUs0/1/3, at least4GiB free with smoke-measured gate;
GPU2 and existing jobs untouched. Container UID:GID1009:1020, existing approved
image with tag/digest recorded. Preserve failures; at most one OOM retry.

## Verification and reporting

Validate historical sources, complete source runs, split/input hashes, stage
lambda1 records and checkpoint metadata. Record cps4 checkpoint and derived
cps2 initialization hashes. Require continuous-field preservation at native
geometry within max1e-4/rms1e-5 voxel tolerance. Unit-test CLI, 60 unique names,
schedule, lambda1 objective/gradients and incomplete-report rejection. Run
rank0 one-step lambda0.1/5 smokes including canonical metrics before production.
For fresh lambda1, compare initial objective to historical cps2 state0 and
report final field/metric reproducibility without requiring bitwise equality.

Require60 finite [160,224,192,3] serialized fields, unchanged prefix-stage
records, and best-loss checkpoint validation before aggregation. Report
Dice/HD95/symmetric ASD, fixed-raw-positive foreground SDlogJ (positive J)
and folding (J<=0, percent); per-case and mean/sample SD. References are the
original842 lambda1 and original VFA/SITReg/FeDDA on the same ten pairs.
Paired deltas compare original842 lambda1 and fresh cps2lambda1. Include
best/stopping epochs, runtime and GPU; save CSV/JSON/Markdown and acceptance.
This is exploratory test-case tuning; timing is diagnostic under shared load.
