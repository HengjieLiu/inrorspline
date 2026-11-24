---
doc_type: codex_execution_spec
status: approved
---

# Rank0 diffusion(lambda1) plus analytic KL-C1 gamma sweep

## Authorization and scope

User requested first test case only, diffusion lambda fixed to1, and gamma
0/1e-4/1e-3/1e-2/1e-1/1. The final1 is10^0, distinct from gamma0.
No first10 extension, adaptive gamma expansion or new regularizer family is
authorized. Complete this pilot and report before choosing a population study.

Carry forward both FD and analytic diffusion baselines from P84. Use analytic
KL-C1 in both combinations because P84 found substantially better consistency
between central/analytic folding and NDV for analytic KL. This derivative choice
is an explicit follow-up default, not a test of every possible FD/analytic mix.

## Objective and frozen method

```text
total = NCC9 + (1 * diffusion + gamma * analytic_KL_C1)
gamma = 0, 0.0001, 0.001, 0.01, 0.1, 1
diffusion backend = FD or analytic
KL epsilon = 0.05; kappa = 120
```

All weights apply unchanged at every stage. Regularize the total current field,
not a new increment. Both regularizers are unmasked on their established domains:
diffusion uses its native full/valid-stencil reduction; KL uses the crop2 total
Jacobian determinant det(I+Du), with analytic derivatives in stage voxel units.
For analytic diffusion, share first derivative fields with the KL calculation.
At gamma0, bypass KL completely to preserve the diffusion-only graph.

Freeze the accepted P84 protocol: seed3 shuffled rank0, fixed0333/moving0115;
raw stored float32 images, squared local NCC9, border image sampling,
endpoint-aligned continuous cubic cps8/4/2 with pyramid4/2/1 and exact dyadic
knot insertion. Start every positive-gamma registration independently at identity.
No optimized diffusion-only prefix is used as initialization.

Fresh Adam each stage; LR0.03/0.01/0.003; max400/800/1300, min40/80/130,
requested patience20/35/50, effective patience>=25, median/check25 and relative
total/data thresholds1e-4/5e-5. Retain the lowest stage total-loss checkpoint,
including initialization. No within-stage LR reduction. Record capped stages.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off; record requested
and effective values. CUDA sampling determinism caveat remains unchanged.

## References and execution

Gamma0 reuses P84's accepted diffusion lambda1 rank0 cold/warm outputs from each
backend. Validate source/protocol/input identity, acceptance markers and field/
model hashes; record exact source paths and reuse receipts externally. Preserve
the historical artifacts. Reused timing is explicitly historical, not measured
again in this execution.

Ten new positive-gamma settings, each rank0 cold followed by rank0 warm in one
fresh worker, give20 new production registrations. Two gamma0 references each
retain their existing cold/warm pair:24 reported observations in12 settings.
Warm rank0 is the primary scientific endpoint; cold metrics provide repeat QA.
Do not choose whichever cold/warm result has higher Dice.

Use all visible GPUs, at most one job perGPU, in the validated experiment runtime
from private configuration. Fix six intra-op and one inter-op/BLAS thread per
worker. Keep registration and evaluation phases separate, with same-GPU replay.
Check memory/storage admission, preserve failed attempts, and do not disturb
unrelated processes. Prior sources and runs remain immutable.

Before production: combined-loss unit/gradient/component tests; gamma0 value,
gradient and optimizer parity for both diffusion backends; baseline hash audit;
four native-volume one-update smokes at gamma1e-4/1 for each backend; independent
saved loss/field/metric replay and PID-attributed telemetry validation.

## Metrics and reports

Retain all35-label Dice/HD95/area-weighted symmetric ASD with eligibility/reasons;
full/foreground FD and analytic BE/diffusion; both determinant conventions'
positive-J SD(logJ), folding, coverage, min/quantiles/mean, J<epsilon fractions
and KL-C1 values; canonical NDV. Foreground is fixed raw image>0. No clamping or
metric redefinition. Undefined values keep explicit counts/reasons.

Log image, diffusion and KL components separately at every evaluated state,
including gamma-weighted KL and both coefficients. Keep best and terminal
checkpoints, optimizer state, histories and source/data/runtime hashes.

Report cold and warm rank0 registration time, startup/load/stage/DVF costs,
allocated/reserved GPU memory,200ms process NVML/RSS peaks, and separate scoring
time/memory. Do not average cold/warm into a population SD or interpret a reused
baseline timing difference as a controlled speed result.

For each backend report every gamma and its signed change from gamma0; compare
the two diffusion backends at each gamma. Select best Dice, then ASD/HD95 and
smaller gamma for exact ties, while also showing accuracy-regularity trade-offs.
No confidence intervals across cases are justified by this single-case pilot.

All generated artifacts stay under `${ARTIFACT_ROOT}/runs/<new-execution>/` and
`${ARTIFACT_ROOT}/repo_experiments/oasis_rank000_diffusion1_analytic_kl_c1_gamma_v1/<new-execution>/`.
The repository retains only sources/specification/log. No commit, sync, paper
archive change or first10 launch is part of this task.
