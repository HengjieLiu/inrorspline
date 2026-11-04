# P74: first10 refined B-spline SVF lambda sweep

Approved280 fresh registrations: ranks0–9, squared local NCC9, refined single
SVF, velocity/displacement diffusion, schedules842/8421, alpha0.01/0.1/0.3/
1/3/10/100 at EVERY active level. No historical fit, shared prefix, cross-lambda
warm start, stage-map composition or extra tuning. Preserve accepted P73 sources.

Unbounded normalized xyz cubic velocity controls, exact dyadic knot insertion,
native160x224x192 seven-step scaling/squaring (v/128), align_cornersTrue,
trilinear border displacement/image sampling, unclamped coordinates. Diffusion
uses total velocity or integrated total displacement, forward-valid unmasked
component/axis-mean squared differences in current loss-grid voxel units.
No image normalization, analytic regularization, dense refinement or tanh.

Pyramid4/2/1/1; LR0.03/0.01/0.003/0.001; max400/800/1300/2000,
min40/80/130/300, patience20/35/50/100 with effective minimum25.
Median/check25, relative total/data1e-4/5e-5. Fresh Adam each stage; cps1
allows two halvings at least100 updates apart. Lowest-total-loss state includes
initialization. Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32False.

Four full-size smokes: both penalties x lambda0.01/100, one update at each of
four levels. Verify lambda1 values/gradients against immutable pilot, lambda
propagation, all case IDs, unique CLI/names, knot/integration/serialization,
absent labels, history and independent replay. Reintegrate all production
outputs at8/10 steps with no optimization; seven steps remain primary.
Record initial scaled-map corner derivative bounds and numerical inverse
residuals. Folding/bound violations are outcomes, not rejection criteria.

Use all available GPUs0–3, one new worker or audit per GPU, UID:GID1009:1020.
Live admission max(8GiB,smoke peak reserved+2GiB); leave concurrent jobs
untouched. Keep cases GPU-affine where practical, retry OOM once after slot
release on the freest eligible GPU. Preserve failed attempts; numerical
verification failures stop dispatch pending audit. Check free artifact capacity.

Save optimizer, worker, metrics/diagnostics and independent-audit time separately;
process PyTorch peak allocated/reservedGiB and stage peaks; GPU identity/free
memory/utilization/process snapshots at launch and during controller execution.
Summarize mean/sampleSD/min/max per setting/lambda; DL timing/memory n/a.
Worker memory excludes other processes; device load must never be attributed
to the registration process. Runtime is diagnostic under concurrent load.

Artifact key:oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1.
Everything generated goes under ARTIFACT_ROOT/runs or repo_experiments.
Require280 finite complete outputs,280 independent audits,980 verified stage
states and840 integration evaluations. No partial ten-case aggregate accepted.
Report four lambda tables with original VFA/SITReg, per-case/per-label CSV/JSON,
paired changes, Dice-first and five-metric Pareto summaries, and line-only2x3
ARC plots (accuracy vs FG SDlogJ above, accuracy vs FG folding below).

No across-case generalization claim: first10 has informed exploratory tuning.
Equal lambda on different regularization targets is not matched constraint
strength. Numerical SVF exponentials do not guarantee zero folding.

Approved validation-only recovery: preserve17 accepted fits and all failure
artifacts; restart the single rank3/lambda0.01 velocity842 failure from identity
in attempt1. No optimizer/protocol edits. A normalized-field allclose failure
requires max<=0.001 voxel and RMS<=1e-5 voxel plus independently recomputed
FP64 knot/velocity preservation<=1e-9 and exponential preservation<=1e-8 voxel.
Keep the original velocity gate. Separate recovery sources/manifests and
worker/independent-audit diagnostics document the correction. Require source
parity, preservation/resume tests, actual failed-checkpoint replay, and a new
four-stage smoke before resuming all four GPU queues.

User-approved execution amendment: increase concurrency to two sweep processes
per GPU0–3, eight total. Replace only the controller and adopt active workers/
auditors without interrupting registration. Preserve all accepted runs and
failure artifacts; numerical worker/audit sources and their hashes remain
unchanged. Each GPU slot covers optimization or independent replay. Require
fresh free memory>=8GiB per admitted process, reserve admission capacity for
still-loading workers, and never exceed two sweep processes on one GPU.
Retain case affinity when feasible; log every actual placement. Record the
concurrency transition and qualify mixed-concurrency timings in the report.

## Approved continue-on-error amendment (2026-10-15)

Supersedes the fail-fast dispatch policy above, not the numerical validation.
Preserve35 independently accepted configurations and all failed attempts. Defer
the rank2/cps8421/displacement/lambda0.01 transition failure until the remaining
244 configurations have been attempted. No new transition tolerance or numerical
retry is authorized. Failures affect one rank/depth/penalty/lambda, not its entire
case or later queue entries. Use two worker/auditor processes perGPU0–3.

Add isolated orchestration/report adapters. Record per-configuration worker,
auditor, parent-validation, launch and interruption failures. Durable OOM budget:
one retry after all ordinary work, selected on a released eligible GPU with most
free memory. Only the current invocation log determines OOM classification;
archive existing independent-audit artifacts before an audit OOM retry.

Use the shared controller lock, durable pre-spawn intent, PID/start-time/UID
identity, and filesystem/process reconciliation on restart. Accepted attempts
win over older failed attempts. Unexplained inactive partial attempts are
preserved/deferred. Never overwrite accepted numerical artifacts. Retain a full
pre-resumption file-hash inventory and verify it after queue termination.

Malformed/missing progress JSON is diagnostic, not a scientific failure. Invalid
GPU telemetry quarantines admissions to that device, not healthy GPUs. Hold an
additional8GiB conservative reservation per active process across polling cycles
until it exits, even though its actual usage may already appear in device memory.
Host available RAM>=32GiB and artifact free capacity>=400GiB gate admissions.
Shared source/provenance/storage failures pause new admissions and are explicit
in status; do not relabel every waiting registration as failed.

Watchdog: warn after30 minutes without CPU/IO/file activity, terminate only the
identity-verified sweep child after60 minutes, allow60 seconds before SIGKILL.
Keep its slot occupied until actual exit, including D-state. This does not detect
every logical hang with continuing activity. Interruptible controller I/O has
deadlines; uninterruptible kernel/NFS stalls cannot be reliably solved by Python.

Queue outcome and scientific acceptance are separate. Failed cells count as
processed, not successful, and do not remain in ETA. Produce failure ledger,
coverage and partial per-case reports; calculate means/SD only for complete
unique ten-case groups. Missing paired settings remain explicitly unavailable.
Never write full-cohort acceptance for incomplete science. Preserve report errors
separately after queue completion; they cannot block subsequent registrations.

## Approved cps1 continuation, validation policy v2

The queue finished278 accepted/two deferred. User approved recovery of ranks2
and5 only, both8421/displacement/lambda0.01, from each failed attempt's own
selected cps2 checkpoint. Preserve all accepted fits, failed attempts and the
coverage-aware report. No prefix optimization is repeated and no cross-cell
initialization is introduced. Fresh Adam and the original full cps1 schedule
remain mandatory; objective, seed, FP32 controls, NCC9, diffusion and seven-step
integration are unchanged.

Validation v2 applies only at cps2-to-cps1. Retain finite/shape checks, prescribed
FP32 knot controls and the original velocity allclose. On original field-gate
failure, require ideal FP64 velocity preservation<=1e-9 voxel and integrated
field<=1e-8 voxel, plus actual rounded controls evaluated with FP64 kernels and
integration: maximum<=1e-4 voxel/RMS<=1e-6 voxel. The FP32 transition remains
bounded by maximum0.002 voxel/RMS1e-5 voxel. Other stages retain v1 exactly.
No segmentation metric selects a checkpoint or authorizes a failed proof.

Add isolated helper/runner/tests. Delegate unchanged original worker/auditor
code through a process-local adapter that copies and verifies completed prefix
checkpoints/history instead of constructing prefix Adam optimizers. Record
prefix hashes, original GPU, new GPU and both worker portions. Report total
optimizer time as original prefix plus new cps1 optimization; worker time is
original failed-attempt duration (one-second timestamp precision) plus new
continuation execution, excluding idle downtime. Keep new updates separately.

Require native-size prefix replay, FP64 proof and one-update/checkpoint/DVF/
canonical-metric fixture for each failure before production. Then run two cps1
continuations on available GPU slots, with fresh free memory>=8GiB or fixture
peak+2GiB. Preserve any failure and allow the other recovery to finish. Follow
with independent four-stage replay and7/8/10-step diagnostics. Only then publish
a complete280-fit report in a distinct continuation_v2 directory and verify the
preexisting artifact inventory. Old outputs remain immutable. High folding at
lambda0.01 is a measured result, not something this validation fix corrects.
