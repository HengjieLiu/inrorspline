---
doc_type: codex_execution_spec
status: approved
---

# First10 diffusion(lambda1) plus analytic KL-C1

## Authorization and fixed scope

The user authorized extending P85 to the seed-3 shuffled first10 with gamma
0/1e-4/1e-3/1e-2/1e-1/1. The final1 is10^0, distinct from zero. Preserve both
FD and analytic diffusion, with analytic KL-C1 throughout. Use all available
GPUs, at most one worker perGPU, and report all accuracy/regularity metrics and
cold/warm time/memory. No automatic gamma extension or more than10 cases.

## Frozen numerical protocol

Objective at every stage: squared local NCC9 +1*diffusion +gamma*analytic_KL_C1.
KL epsilon0.05/kappa120; crop2 det(I+Du) of the total current spline in stage
voxel units. Diffusion retains its P84 full/valid-domain mean. Analytic
diffusion shares its first derivatives with KL. Both losses are unmasked.
Use P85's exact CombinedObjective and component-aware optimizer, not a new
numerical implementation. Preserve all P84/P85 numerical source files.

Prolonged cubic cps8/4/2; endpoint-aligned pyramid4/2/1; exact dyadic knot
insertion; zero/identity start independently for every registration. Raw
float32 stored images; border image sampling and canonical nearest-label warp.
Fresh Adam per stage, LR0.03/0.01/0.003; max400/800/1300, min40/80/130,
requested patience20/35/50, effective>=25, median/check25, relative total/data
thresholds1e-4/5e-5; select the minimum stage total objective including initial
state. No within-stage LR reduction, as in P84/P85. Record capped stages.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off; record effective
flags. Six intra-op and one inter-op/BLAS thread per worker.

## Dataset, reuse and execution

Use `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`, ranks0..9,
with the first element fixed and second moving. Native stored160x192x224,
reoriented160x224x192,1mm. Foreground is fixed raw image>0; labels are used
only for posthoc accuracy. Dataset paths and runtime mounts remain in private
configuration. Use the validated pinned-image experiment runtime from P84/P85.

Gamma0 reuses both accepted P84 diffusion lambda1 cohort blocks, all11
observations each: cold rank0 and warm0..9. Audit source/protocol/pair/input/
field/model hashes and metric receipts. Keep exact historical provenance and
time/memory. Positive-gamma rank0 is freshly rerun for each cohort worker;
do not splice P85 timing into a new cohort. Its field can be checked against
P85 as numerical QA. Never alter historical artifacts or source snapshots.

Ten new settings give110 production registrations (cold0 then warm0..9).
With22 reference observations,12 settings yield120 warm scientific endpoints.
Cold rank0 is repeat QA, not an eleventh case or independent subject.
Registration and independent evaluation run in separate waves on the same
GPU for each setting. Preserve failed attempts and source snapshots. Monitor
exact host PID/start identity, NVML and RSS every200ms. Require memory/disk
admission and sufficient measured artifact projection with30% margin.

Preflight:19 P84 tests,5 combined-loss tests,2 first10 scope/adapter tests;
audit both complete gamma0 reference cohorts; four native-volume one-update
smokes covering both diffusion backends and gamma1e-4/1; independent loss,
field and full-metric replay plus attributed telemetry.

## Metrics, analysis and artifact roots

Retain all35-label Dice, HD95 and area-weighted symmetric/directed ASD,
eligibility/counts/exclusion reasons, and image loss. Retain full/FG FD and
analytic diffusion/BE in voxel and normalized units; central/analytic J
positive-only SDlogJ, folding, coverage, min/quantiles/mean, J<epsilon and
KL-C1; raw and normalized canonical NDV. Record every stage's selected and
actual updates, stopping, data/diffusion/raw/weighted KL components and best/
last checkpoints. Do not clamp SDlogJ inputs or redefine NDV.

Primary summaries use10 warm cases per setting, means and case-level sample
SD, not pooled voxels or labels. Compare each positive gamma with matching
backend gamma0 case by case, and analytic-minus-FD diffusion at matching gamma.
Report signed paired mean/median changes, wins/ties/losses and exploratory
95% paired case-bootstrap intervals (20000 resamples, fixed seed). These are
small exploratory test-set tuning results, not held-out confirmation; no
multiple-testing-adjusted significance claim. If label eligibility differs,
provide common-label sensitivity alongside canonical accuracy.

Report every setting's cold rank0 and warm mean/sample-SD/range time, stage/
startup/load/DVF costs, Torch allocated/reserved and sampled process GPU/RSS
peaks. Scoring has a separate clock/resource scope. Gamma0 metrics are paired
scientifically, but historical resource comparisons are observational and may
mix physical GPUs/update counts. Select best mean Dice, then ASD/HD95/lower
gamma on ties; show regularity trade-offs instead of a universal winner.

Source and manual notes live here and in `scripts/experiments/oasis_diffusion_kl_first10/`.
Generated outputs: `${ARTIFACT_ROOT}/runs/<execution>/` and
`${ARTIFACT_ROOT}/repo_experiments/oasis_first10_diffusion1_analytic_kl_c1_gamma_v1/<execution>/`.
The independently versioned posthoc analysis source is
`scripts/reports/analyze_oasis_diffusion_kl_first10.py`; record its hash and
input hashes in the external analysis receipt. No commit or sync is requested.
