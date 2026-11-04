# Technical log

## Known limitations

First-ten test-case tuning is exploratory. Deterministic cuDNN does not ensure
globally deterministic CUDA grid_sample backward. Historical outputs remain
read-only; shared-workload runtimes are diagnostic. Prefix stages retain
evaluation-grid voxel diffusion units, while cps2 uses full-resolution units.

## Audit

The saved original842 runs retain stage1/best.pt, including cps4 controls
[44,60,52,3] and the exact selected stage metadata. New fits branch here, not
from the old cps2 terminal field. Dyadic insertion produces [84,116,100,3]
controls. Existing stage optimizer/objective/metrics are imported unchanged.
New manifests separate stage weights [1,1,cps2_alpha] and hash the cps4 source
and prolonged initialization. The old842 field hash is explicitly source
provenance, not falsely described as the cps2 initialization.

GPUs0/1/3 had approximately48.5GiB free and zero utilization at audit; GPU2
had an unrelated workload and is excluded. Four new unit tests passed before
the smoke launch. Smokes and production remain separate phases.

Both rank0 one-step smokes passed at 2026-10-13 14:14:37 PDT. Prolongation
max/RMS differences were 4.98444e-6/1.67023e-7 voxel. CPU/GPU-derived initial
control hashes matched exactly. Native NCC9/diffusion values and gradients
matched exactly; peak tensor allocation was 1.46116 GiB, retaining the4GiB
free-memory gate. All16 historical tests also passed (20 tests total).
Production launched after inspecting these gates, on GPUs0/1/3.

## Completion and acceptance

All60 fits completed at 2026-10-13 14:31:07 PDT. No failed attempts or OOM
retries (62 successful attempts including two smokes). Cps2 stopping epochs
ranged175–625. Mean optimization/total-worker time was25.8829/39.5586s;
the approximately13.68s difference covers initialization/loading, evaluation,
serialization and other worker overhead, not additional optimization.

Independent read-only checks passed for accepted report hashes, 60 candidates,
100 comparison rows, ten complete groups, across-case sample SD, canonical
per-label aggregation and absent-label consistency, unchanged cps8/cps4
records, shared prolonged initialization hashes and paired metric deltas.
All ten freshlambda1 fields have identical hashes to original842lambda1;
all five metric differences are exactly zero. This is observed reproduction,
not a general guarantee of bitwise CUDA determinism.

Cps2lambda0.5 has the highest mean Dice0.829577 and lowest ASD0.350710mm;
lambda1 has the lowest HD951.787126mm. Relative tolambda1, lambda0.5 changes
Dice +0.003281, HD95 +0.019762mm, ASD -0.005604mm, FG SDlogJ +0.088096 and
FG folding +1.540182 percentage points. Lambda5 lowers folding to0.497181%
but reduces Dice to0.797722. No weight matches VFA/SITReg folding while
retaining their accuracy. These are exploratory first-ten comparisons.

Complete CSV/JSON/Markdown, all per-case deltas and thelambda1 reproducibility
appendix are in the external report directory under the experiment artifact
key, with ACCEPTED.json written after all60 outputs and report checks passed.
