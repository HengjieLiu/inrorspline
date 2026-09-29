---
doc_type: codex_execution_spec
status: completed
---

# MR-D-BSCP 8-4-2 acceleration: GPU 0, first round

## Frozen protocol and evidence limits

Authorized scope is the audited first implementation round: a reference/correctness
harness, isolated kernel caching, grouped-to-batched expansion, fixed NCC statistics,
buffered loss history, and numerically gated control-space BE; then combination and
first10 confirmation. Later compiled kernels, FireANTs, mixed precision, and new
optimizers are deferred until this report is reviewed. No paper edits.

User amendment during implementation: test control-space BE even if the strict
FP32/Adam numerical gates fail. Include it as an explicitly non-equivalent
exploratory registration arm, quantify Dice and HD95 as well as time/memory and
DVF/Jacobian differences, and include all first10 cases. Numerical gate failure
remains visible; it is not silently waived or treated as proof of clinically
meaningful degradation. Strict numerical promotion and empirical registration
quality are separate decisions. HD95 uses the repo's existing label/surface
convention and is computed outside timing. No HD95 acceptance tolerance was
preregistered: report paired absolute changes, including per-label changes.

Reference: MR_A_fast_decay, alpha 1000, optimizer seed 1, seed-3 shuffled OASIS
ranks 0-9, full cps 8/4/2, downsample 4/2/1, stage LR .03/.01/.003,
max steps 400/800/1300, minimum 40/80/130, patience 20/35/50, checks 25,
relative total/data thresholds 1e-4/5e-5. Preserve original NCC/custom backward,
crop alignment, BE units, Adam defaults and stage resets.

GPU: physical 0 only, sequential work. cuDNN benchmark=False, deterministic=True.
Preserve and record matmul TF32=False, cuDNN TF32=True, deterministic algorithms=False.
Default artifact-store Docker environment and owner UID/GID; resolve private paths
from local configuration. Six CPU threads per numerical worker.

## Source and artifact layout

Source implementation: scripts/experiments/mrdbscp842_acceleration/.
Run and review roots: ARTIFACT_ROOT/runs/mrdbscp842_acceleration_gpu0_<UTC timestamp>
and ARTIFACT_ROOT/repo_experiments/2026-09-19_mrdbscp842_acceleration_gpu0/<timestamp>.
All fields, controls, profiles, source snapshots, manifests, checks and reports are external.
Record HEAD, dirty diff, hashes and copies of every relevant source, split, config,
container image ID, versions and actual commands. Never overwrite a prior run.

## Gates and sequence

1. Static checks, CPU algebra/behavior checks, manifest-only dry run, GPU operator checks.
2. Original-reference one-step smoke and candidate smoke only after operator correctness.
3. Reference A/A: rank 0 cold plus five warm repeats. Require stable stopping and
   field/quality parity; report robust runtime spread, target <=3%.
4. Baseline first10 (fresh worker cold rank 0, then warm ranks 0-9), plus separate
   diagnostic control checkpoints and a profiler run. Five states per stage:
   initial, first update, midpoint, 25 steps before stop, final.
5. Single-factor variants: cached kernels, batched expansion, cached fixed NCC
   statistics, buffered history, stable factored control-space BE. Invalid or
   numerically unstable candidates are excluded from strict promotion; the user
   explicitly authorized BE registration characterization despite its failed gate.
   Negative/no-gain results are retained.
6. Microbenchmark: three real stage shapes, forward+backward with new graphs;
   10 warmups and 5 groups of 25 calls. Target module median improvement >=10%
   (>=20% for BE). Full registration may also be measured for cheap variants
   to assess setup/host costs not visible in operator time.
7. Pilot: ranks 9/0/7, three paired repetitions, fixed-reference steps and original
   adaptive mode, 36 primary measurements per reference/candidate comparison.
   Alternate AB/BA worker blocks. Each block has its own uncounted warmup.
8. Combine only correctness-passing variants with observed benefit; rerun key gates.
   The automatic selection rule is conservative: combine the isolated non-BE
   variants that pass all preregistered gates, including the >=5% runtime gain
   and paired-bootstrap criterion in both fixed and adaptive modes. This rule
   was recorded after the cache pilot but before the other pilot outcomes.
   BE remains a separately reported exploratory arm, as explicitly requested.
9. Confirm winner against B0: first10, five paired repeats, both modes = 200
   primary rows, plus three fresh-worker rank-0 cold measurements per backend.
10. Separate evaluation of saved fields; final machine-readable decisions and Markdown
    report include failure evidence, timing/quality/memory, source and replay commands.

The exploratory BE arm additionally receives first10 fixed/adaptive paired
characterization even if it is not eligible as the strict-equivalence winner.
It may be compared or combined experimentally with passing infrastructure
optimizations, but any such combination keeps the BE non-equivalence warning.

Performance: per-case repeat medians; paired candidate/reference ratios; 10,000
case-cluster bootstrap draws with seed 20260919. Speed pass <=.95 ratio and
95% CI upper <1, >=90% cases <=1.02, disclose any >1.10. Memory-only pass
>=20% allocated reduction and runtime ratio <=1.02. Default memory increase cap 10%.

Strict output gates: float32 finite XYZ normalized DVF (160,224,192,3);
vector RMS <=.005 voxel, vector max <=.05 voxel; case mean Dice delta <=1e-4,
per-label <=1e-3, same missing-label pattern; nonpositive-Jacobian fraction
increase <=1e-5. Match stage steps and structured stop category; relative-improvement
numbers remain available, not compared as byte-identical formatted reason strings.
No silent tolerance changes. Unstable reference => inconclusive strict promotion.

## Timing and monitoring

Headline time remains case_load_complete to dvf_file_ready including stage setup,
optimization, final expansion, CPU transfer and completed NPY write. New endpoint-only
instrumentation is identical for both sides. Deep/event/profiler runs are separate,
with an instrumentation bridge to the historical timing mode. CPU wait and GPU
phase intervals overlap and are not added as disjoint costs.

A host controller identifies exact worker host PIDs using /proc command arguments,
process start times and a per-worker token. A container-side worker waits for explicit
host identification before CUDA work. Only that process and verified descendants
are allowed. Poll the GPU approximately every second plus interval boundaries;
monitor/query failures fail closed. Retain contaminated attempts, rerun the affected
warm case after idleness, restart a fresh worker for contaminated cold work.
Polling cannot exclude external jobs shorter than its sampling interval.

## Numerical tests

Use small independent FP64 sampled-matrix oracles because repo BE forces FP32.
Compare actual repo FP32 expansion/BE/VJP and one-/25-step Adam, on zero, constant,
affine, near-affine, random and boundary-impulse fields. Test true stage shapes and
diagnostic controls. Ordinary gradients relative L2 <=1e-4; expansion rtol 3e-5,
atol 3e-7; weighted BE absolute difference <=1e-7+3e-5*abs(reference).
Nullspace gates include FP64 energy/gradient and regularizer-only update drift.
Test source stopping slices, loss-buffer row identity, coordinate convention,
and deliberate axis/crop/mixed-factor mutations.

Stable CP-BE prototype: factor derivative evaluations into finite control differences
and lower-degree spline evaluation; reduced QR of the finite 1-D evaluation yields
a PSD square-norm expression. No diagonal jitter, loss clamp, continuous-integral
replacement, or giant 3-D matrix. Dense control-axis factors are a first feasibility
implementation; speed is measured, not assumed.

## Replay interface

The host suite accepts explicit --container, --repo-container, --artifact-host,
--artifact-container, --oasis-root, --gpu 0 and --run-id; --dry-run emits a
manifest without launching workers. A run manifest records the exact invocation.
Workers, validation and report commands are also independently replayable.
