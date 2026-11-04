# Rank0 cps1 B-spline versus dense after [1,1,5] prefix

## Requested scope and fixed settings

One fresh production fit on seed3 shuffled OASIS rank0: fixed0333, moving0115,
moving-to-fixed registration. Use the accepted P61 cps8/4/2 prefix with
diffusion weights [1,1,5]. Prolong saved cps2 controls once by dyadic cubic knot
insertion to cps1 controls [164,228,196,3]. Optimize these controls with final
lambda1. Compare against the accepted P62 dense [1,1,5,1] result, not a mean
over cases. Preserve all previous runners and artifacts as read-only.

The only intended experimental difference is final-stage representation.
Retain unnormalized local squared NCC9 (1-mean(correlation²)), raw intensities,
border sampling, align_cornersTrue and full-volume unmasked finite-difference
L2 diffusion of the expanded total voxel-unit displacement. This is an
explicit override of the default analytic B-spline regularizer; do not change
to analytic diffusion or bending energy. Both parameters use normalized XYZ.

Prefix LRs0.03/0.01/0.003 and downsampling4/2/1 are inherited unchanged.
Fresh final Adam, LR0.001, max/min2000/300, check/window25, patience100,
relative total/data thresholds1e-4/5e-5, two LR halvings with100-epoch spacing.
Select the lowest-total-loss state, including initialization, never labels.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off.

## Verification and execution

New isolated runner cps2lambda5_cps1.py; unit tests cover CLI scope, schedule
parity, names, synthetic prolongation, objective equivalence and gradient
chain rule. Validate accepted reference report, source/input/checkpoint hashes,
case, direction, stages and split. Require cps2 reconstruction and cps2→cps1
initial-field errors below max1e-4/rms1e-5 voxel; record actual differences.
Compare initial total/data/reg against the dense objective and saved run.
One-step smoke through Adam, field serialization, native loss/gradient parity
and canonical metrics gates the full fit. Require finite [160,224,192,3] field,
saved control checkpoint, correct best-loss state and completion marker.

GPU3 sequential smoke then production; UID:GID1009:1020. Record live memory
and utilization, image/digest, exact source hashes, Git revision and branch.
Leave all existing jobs untouched, especially GPUs0/2 workloads. Require
4GiB free or measured smoke peak+2GiB, whichever is larger. Preserve any
failure and stop for audit; no uncontrolled restart or new sweep.

External run/report key under ${ARTIFACT_ROOT}/runs and /repo_experiments:
oasis_seed3_rank000_mrdbscp8421_lambda1151_vs_dense_v1.
CSV/JSON/Markdown include cps1, saved dense, shared prefix, original
VFA/SITReg/FeDDA for rank0: Dice, HD95, symmetric ASD, fixed-raw-positive
foreground positive-J SD(log J), J<=0 folding percent and paired differences.
Record best/stopping epochs and final-stage runtime. No across-case SD for n=1.

## Interpretation limits

One exploratory case, no generalization or statistical superiority claim.
B-splines are smooth but do not guarantee positive Jacobians. Identical
nominal Adam schedules do not imply identical field-space updates between
representations. Existing GPU load makes timing diagnostic only; CUDA
grid_sample backward can remain nondeterministic. No a priori closeness
margin was requested, so report the actual paired differences.
