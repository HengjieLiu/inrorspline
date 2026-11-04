# First10 refined-SVF lambda sweep

P74 adds isolated `bspline_svf_sweep.py`, `run_oasis_bspline_svf_sweep.py`,
and `report_oasis_bspline_svf_sweep.py` under
`scripts/experiments/oasis_alpha_sweep/`. P73 sources remain hash-frozen.
The numerical objective subclasses the audited P73 objective and explicitly
sets its alpha; the native exponential, exact knot insertion, raw NCC9,
evaluation-grid voxel diffusion and state-indexed Adam are unchanged.

The matrix is280 independent fits: ten shuffled cases x seven global stage
weights x two penalties x two depths. No prefix sharing, implicit warm starts,
rank0-only filenames, or lambda1-only histories are permitted. A configuration
hash, case rank, depth, penalty and weight identify every new run.

Each GPU slot covers a worker followed by a separate independent replay.
Case affinity limits hardware confounding; OOM retry may use a released GPU.
Source/input/reference hashes, canonical absent-label rules and complete
ten-case groups gate reporting. Seven-step fields remain primary;8/10 steps
are diagnostic reintegrations, not new optimization or selectable outputs.

## Time and memory

- `optimizer_sec`: sum of stage optimization intervals, including objective
  evaluation/checkpoint bookkeeping inside the historical optimizer timer.
- `worker_sec`: loading, optimization, stage serialization, metrics and
  integration/topology diagnostics, excluding independent replay and queue wait.
- `metrics_diagnostics_sec`: final canonical metrics and7/8/10 diagnostics.
- `audit_sec`: independent checkpoint/loss/history/metric replay process time.
- `peak_cuda_gib` / `peak_reserved_gib`: maximum worker PyTorch allocated /
  allocator-reserved memory. Stage timers reset peak counters, so the worker
  explicitly takes the maximum across stages and final diagnostics.
- `audit_peak_*`: separate replay-process peaks. They are not added to worker
  peaks because worker and audit run sequentially within one slot.

Neither Torch memory measurement includes other processes; it is not equivalent
to device-wide nvidia-smi usage or every CUDA context allocation. Device snapshots
at launch/end and30-second controller samples record the concurrent context.
Admission requires max(8GiB,largest smoke worker/audit reserved peak+2GiB).
Per-setting/per-lambda reports include mean/sampleSD/min/max; four pooled method
resource rows explicitly combine seven lambdas. Timing is shared-load diagnostic,
not an isolated benchmark; original VFA/SITReg resource values are n/a.

## Validation-only recovery

`run_oasis_bspline_svf_sweep_recovery.py` preserves the original runner and all
its source hashes. AST regression verifies its optimization worker differs
only in transition checking and provenance. `svf_transition_validation.py`
keeps the original velocity/field allclose fast path. A field-only rejection
may pass only with max<=0.001 voxel, RMS<=1e-5 voxel, exact FP32 knot controls,
and FP64 re-expansion/re-integration agreement (velocity<=1e-9 voxel,
field<=1e-8 voxel). FP64 results are never used for optimization or selection.
Independent replay repeats the same validation. Invalid/nonfinite values or
larger differences still stop dispatch; folding is still a reported outcome.

The separate recovery manifest hashes its sources, failed-transition proof,
and preservation inventory. Completed runs are skipped, only the explicitly
approved failure is restarted from identity in a new attempt, and failure
logs are retained. Fresh recovery smoke plus independent replay gates resume.
Final reports include a recovery appendix and recheck every preserved file.

## Eight-process queue

`queue_oasis_svf_eight_workers.py` is an orchestration-only adapter for the
user-approved two-process/GPU extension. It invokes the immutable recovery
entry point, adopts active children across controller handover, and validates
completed fits without rerunning them. Process PID/start-time checks prevent
PID reuse from being mistaken for a live adopted worker. Optimizers and audits
share the same per-GPU limit. Every launch reserves8GiB admission capacity;
fresh device samples and concurrency events accompany the original per-process
time/memory measurements. Report timings as mixed-concurrency shared-load data.

## Continue-on-error controller

`queue_oasis_svf_continue_on_error.py` supersedes fail-fast orchestration only.
Worker/auditor and transition-check sources stay hash-frozen. It preserves all
accepted results and failures, defers a failing configuration, and proceeds with
unrelated configurations. A durable state journal, shared lock, pre-spawn intent,
process reconciliation, current-log-only OOM classification and one tail retry
make restarts bounded. Progress corruption is nonfatal; partial GPU telemetry
disables only affected device admissions. Conservative8GiB reservations persist
for each active child; CPU RAM/storage/source guards pause shared admissions.
The inactivity watchdog verifies child identity and retains slots until exit.

`report_oasis_svf_queue_outcome.py` reports all accepted per-case metrics,
coverage, deferred failures, available paired differences, stage/integration
diagnostics and time/memory. Only complete unique ten-case groups receive means
and sample SD. Queue completion with failures is not scientific acceptance.
An absent comparison is explicit, not an exception or an implicit cohort change.
Detailed guards, remaining kernel-I/O limitations and approval are in P74's spec.

## Two-case cps1 continuation (v2)

`svf_transition_validation_v2.py` adds a cps1-only precision proof. The original
ideal FP64 knot/integration checks are supplemented by evaluation of the actual
rounded FP32 control values using FP64 spline kernels and integration. This
distinguishes coefficient rounding from FP32 sampling amplification. Exact
prescribed knot controls, the velocity gate, and physical max/RMS caps remain.
The numerical FP32 objective and integration are never replaced by the proof.

`resume_oasis_svf_cps1_v2.py` delegates the immutable recovery worker/auditor
through a temporary process-local adapter. Prefix optimization calls instead
load, hash, replay and copy the already selected same-registration stages;
only cps1 calls the original Adam optimizer. The independent auditor still
replays all four stages, histories, canonical metrics and integration diagnostics.
Rank2/5 continuation identity, original/new GPU, copied-prefix update counts,
new updates and combined timing are explicit. Old reports and278 accepted fits
are preserved; publication uses a separate continuation_v2 report directory.
