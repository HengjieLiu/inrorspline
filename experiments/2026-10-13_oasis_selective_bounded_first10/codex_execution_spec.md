# P67: first-ten selective bounded composition extension

User authorized the four frozen P64/P66 variants on seed3 shuffled OASIS
ranks0–9 and one new worker per available GPU0/1/2/3. Reuse the four accepted
rank0 results without rewriting them; run36 fresh identity registrations.

All variants retain four composed elementary maps cps8/4/2/1. Bound last1,
last2, last3, or all4 increments. Earlier increments remain unbounded c=p;
bounded increments use c=a*tanh(p/a), unit derivative at zero, validated
endpoint-aware0.99 cap. No reuse of an optimized prefix, knot insertion,
accumulated-map resampling, field clipping, added increments or tuning.

Frozen weights[1,1,5,1], loss downsampling[4,2,1,1], normalized latent
LR[0.03,0.01,0.003,0.001]. Raw-intensity squared local NCC9 plus unmasked
total-composition forward-valid diffusion in evaluation-grid voxel units.
This explicitly overrides the repository analytic-regularizer default.
Max/min epochs400/40,800/80,1300/130,2000/300; patience20/35/50/100 with
effective minimum25; check/window25; relative total/data1e-4/5e-5.
Fresh Adam each stage, minimum-total-loss checkpoint including identity.
Only cps1 permits two halvings at least100 epochs apart. Seed1, FP32,
cuDNN benchmarkFalse/deterministicTrue, TF32off; CUDA sampling backward
may remain nondeterministic. No segmentation/reference supervision.

Separate first-ten adapter reuses the historical objective and optimizer;
all historical sources and outputs remain immutable. Case, split, input,
source, checkpoint, field and reference hashes gate reuse and acceptance.
Tests verify four masks,80 unique production/smoke names, ten-case parsing,
legacy all-bounded parity and complete-cohort sample-SD aggregation.
Two full-size rank1 four-stage one-step smokes cover bounding extremes.

Execute as1009:1020, one worker per physical GPU, current admission at least
8GiB or smoke allocated peak+2GiB. Keep each case's four variants on the same
GPU. Existing jobs untouched. Preserve failures, stop scheduling and drain
our already-running workers for audit. No automatic retries or new searches.

Each worker validates four checkpoint/control/loss reconstructions, finite
native160x224x192x3 serialization, canonical label metrics, all native-cell
corner derivatives and100k shared seed1 foreground off-grid probes. Only
bounded elementary maps must satisfy contraction; earlier unbounded folds
are measured, not rejected. Check common unbounded prefixes per case at
loss1e-6, max1e-4/RMS1e-5 voxel tolerance.

Artifact key under external ARTIFACT_ROOT runs/repo_experiments:
oasis_seed3_first10_mrdbscp8421_selective_tanh_lambda1151_v1.
Aggregate only40 successful combinations. CSV/JSON/Markdown: per-case and
mean ± sampleSD(ddof1), paired changes vs all4/VFA/SITReg, stage diagnostics,
stopping/best epochs, runtime/GPU/bound saturation. Five primary metrics:
Dice, HD95mm, symmetric ASDmm, fixed-raw-positive positive-J SD(logJ),
foreground J≤0 percent. Report ranks1–9 separately from ranks0–9 because
rank0 guided exploratory design. No generalization or global topology claim.
