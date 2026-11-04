# P74 technical log

## Limitations

Exploratory first10 tuning, finite-step/interpolated SVF exponential, possible
CUDA grid_sample backward nondeterminism, unequal effective velocity versus
displacement regularization, and timing under concurrent workloads. No implicit
zero-folding guarantee or equal-strength interpretation of lambda.

## Two-case cps1 continuation approval and evidence

The continue-on-error queue ended2026-10-15 16:54:25UTC with278 accepted and
two deferred configurations. Reports finished16:55:37UTC; the1425-file earlier
preservation inventory passed16:57:27UTC. Only8421/displacement/lambda0.01 is
incomplete, missing ranks2 and5;27 of28 settings have complete first10 summaries.

Read-only replay reproduced both saved cps2 fields exactly. Rank2 maximum/RMS
FP32 transition differences:0.0010928381234407425/1.6594131178964186e-6 voxel;
rank5:0.0010101795196533203/1.3231485659020392e-6 voxel. Only one component
per field exceeds0.001 voxel. Ideal FP64 integrated-field differences are
1.198440652272481e-12 and9.484383764468696e-13 voxel. Actual rounded FP32
controls evaluated in FP64 differ by maxima1.687157822993085e-5 and
1.5258434095328025e-5 voxel; RMS7.457556076414766e-8/6.817502541426442e-8.
All three accuracy metrics and total/data losses were unchanged across each
initialization. FG SDlogJ changes were2.73e-5/3.05e-5; folding changes were
-4.73e-5/-4.87e-5 percentage points. Prefix folding itself is8.3778%/8.0007%,
which remains a scientific outcome, not a validation error to remove.

User approved versioned proof-gated maximum0.002/RMS1e-5 voxel validation,
including new actual-rounded-control FP64 max1e-4/RMS1e-6 requirements, and
continuation of only cps1 from the two own-prefix checkpoints. The isolated
adapter delegates the unchanged numerical worker and independent auditor;
completed prefix stages are byte-copied and replayed, not reoptimized. New
continuation provenance identifies the original prefix GPU, hashes and worker
time. Preserve all278 successes and every old failure/report; publish elsewhere.

Eleven new tests passed for limits, mandatory ideal/rounded FP64 proofs, invalid
controls/nonfinite/velocity rejection, unchanged prefix dispatch, real small-grid
FP64 proof, approved scope, adapter restoration, prefix replay without Adam,
final-stage-only optimization delegation and preservation rejection. Full-size
fixtures and complete regression checks gate production.

All61 SVF regression tests passed, including the11 new continuation tests.
Preservation preflight revalidated278 accepted rows and confirmed exactly the
two approved pre-cps1 failures. A full file inventory of existing runs, failed
attempts and reports is being recorded before native-size fixtures. No old
source, checkpoint, history or result has been edited by this amendment.

The inventory contains11605 files. First native fixtures stopped during setup,
before image loading or any optimization: the process-local transition proxy
omitted the legacy POLICY attribute used by the original manifest checker.
Both failed fixture directories, controller/failure logs and the first frozen
manifest are retained. Adapter revision1 exposes that unchanged v1 metadata
while dispatching cps1 verification to the separately asserted v2 policy.
New regression coverage checks this legacy contract in production and fixture
contexts. Revision1 uses a new manifest, fixture directories and progress files;
no failed fixture or old scientific output is overwritten.

Revision1 passed all62 regression tests. Its preservation inventory contains
11619 files including the initial adapter failure artifacts. Both native-size
fixtures passed2026-10-15 19:11UTC: copied prefixes replayed exactly, cps1
passed both ideal and actual-rounded-control FP64 proofs, and one Adam update
passed checkpoint/DVF serialization plus canonical metrics. Peak reserved
memory3.791/3.770GiB; admission stays8GiB. Production cps1 continuation began
19:11:36UTC onGPU3 (rank2) andGPU1 (rank5). At19:12:19UTC both reached150
new cps1 updates with finite decreasing losses. No prefix was reoptimized.

Both continuations passed independent four-stage/loss/history/canonical-metric
and7/8/10-step replay by2026-10-15 19:18:53UTC. Rank2 cps1 stopped at1800
updates(best1800); rank5 at1850(best1849), using the prescribed adaptive
schedule and LR reductions. All280 registrations are now independently valid;
the full preservation inventory and complete report publication remain the
final acceptance gates. The two low-lambda outputs retain substantial measured
folding; this recovery repairs validation, not the regularization tradeoff.

Final acceptance:2026-10-15 19:21:02UTC. All11619 inventoried preexisting files
verified unchanged, including278 accepted registrations, the two failed attempts,
and initial adapter-smoke failures. The separate continuation_v2 report contains
280 independently audited candidates,30 summary rows(28 configurations plus two
DL references),980 stage rows,840 integration evaluations,9800 label rows and1400
paired rows. Every summary uses all ten cases. A separate read-back verified
all generated acceptance hashes, completion markers and new-update counts.
Root progress is accepted280/280,zero deferred. Four sweep tables, CSV/JSON,
line-only ARC figures, resource statistics and recovery/timing disclosures are
published under ARTIFACT_ROOT/repo_experiments/<P74-key>/continuation_v2.

## Implementation audit

P73 hard-codes rank0 and lambda1 in specification, objective, case selection,
paths and histories. P74 introduces isolated generalized sources and leaves
all executed P73 numerical sources and accepted artifacts unchanged. Test actual
lambda scaling of loss/gradient and case metadata, not only CLI arguments.
The historical runtime n/a formatting bug is avoided with explicit nullable
reference resource metrics and report regression tests.

User approved all four GPUs with existing workloads. Initial read-only check
found approximately30–44GiB free per GPU and6.4TiB artifact storage free.
Fresh checks occur at every admission; these initial readings are not reused
as launch-time measurements. Record time and process memory per method/lambda.

## Approved transition-check recovery

The initial controller stopped after17 independently accepted fits. Rank3,
velocity diffusion lambda0.01,842 failed the cps4-to-cps2 integrated-field
transition check. Its two completed stages, failed attempt and logs remain
read-only. This was not a CUDA OOM or nonfinite optimization failure.

Read-only replay reproduced the saved cps4 field exactly. FP32 knot insertion
changed velocity by at most9.969e-6 voxel; subsequent seven-step integration
differed by max6.8401e-4 voxel and RMS7.1669e-7 voxel. Only two components
failed the original normalized-coordinate allclose test. Recomputing the
spline kernels, knot insertion and integration in FP64 reduces the maximum
velocity/field discrepancies to1.238e-14/1.031e-12 voxel. Canonical Dice,
HD95, ASD and foreground folding were identical across the transition;
foreground SD(logJ) differed by-7.499e-7.

User approved auditing/repairing the check, preserving successes/failures,
and resuming. Isolated recovery sources leave all executed numerical sources
unchanged. The original velocity and field checks remain the fast path. A
failed field check additionally requires max error<=0.001 voxel, RMS<=1e-5
voxel, exact prescribed FP32 knot controls, and independent FP64 preservation
of velocity<=1e-9 and integrated field<=1e-8 voxel. This is a validation-only
roundoff allowance, not an optimization precision change or folding filter.
Both worker and independent auditor record/reproduce the transition check.

Recovery creates a separate hashed manifest and full preservation inventory;
retains17 accepted registrations, retries only the approved failed cell from
identity in a fresh attempt1, and queues262 unlaunched cells. Numerical failures
still stop new dispatch. Pending cells are now reported as blocked rather
than mislabeled failed; stopped progress has no ETA. Four historical full-size
smokes remain immutable; one additional four-stage recovery smoke and replay
must pass before restarting production. All four GPUs remain subject to fresh
admission checks and no other jobs are changed.

Recovery verification: all17 original unit/report tests and seven new guard,
AST parity, preservation and resume tests passed. The fresh native-size8421
velocity/lambda0.01 smoke and independent replay passed. Peak reserved memory
was4.166GiB for worker and1.852GiB for auditor; admission remains8GiB. Exact
acceptance timestamp, source hashes and fixture metrics are in recovery_v1.
The detached recovery controller has been launched; completion remains pending.

Four production workers started2026-10-15 12:30:48UTC, one each onGPU0–3.
Ranks0/1/2/3 retain their original GPU2/0/1/3 affinity. The preserved inventory
contains691 files, verified before dispatch and checked again before final
aggregation. The retry uses attempt1; the other three resumed cells use new
attempt0 directories. Fresh free memory was27.1–37.6GiB per admitted GPU.

At12:32UTC all four resumed workers reached cps2 optimization without failure.
The rank3 retry passed the previously failing cps4-to-cps2 transition. Accepted
count remains17/280 at this checkpoint; retry completion and full-cohort
acceptance are not yet claimed. Provisional remaining ETA6–10hours uses earlier
observed worker/audit durations and is conditional on current concurrent load.

## Eight-process concurrency amendment

User requested two tasks perGPU0–3 (eight total) to improve throughput. New
`queue_oasis_svf_eight_workers.py` changes scheduling only and invokes the
unchanged hash-frozen recovery worker/auditor. Controller handover targets only
the exact old controller PID after checking command line/start time/ownership;
active children are adopted and are not signaled. All accepted cells are
validated/skipped and existing attempts are never overwritten.

Both optimizers and audits count against two slots/GPU. Admission checks live
free memory and reserves8GiB for a newly launched process still loading. The
case/GPU preference is retained where feasible, but actual placements are
recorded and idle slots may serve another case. One OOM retry remains allowed;
other failures stop new dispatch. A distinct parallel8_v1 manifest, launch/
GPU logs and final reporting appendix record this mixed-concurrency execution.
No twofold speedup is presumed: shared GPU compute, memory bandwidth and
external jobs may limit throughput. Numerical controls/losses are unchanged.

Six scheduler tests passed: per-GPU/global capacity, low-memory rejection, CLI,
affinity/audit priority, accepted-fit preservation, active-worker adoption,
completion-marker handling and PID-reuse protection. Read-only live preflight
validated all numerical sources,21 accepted fits, and four adoptable workers.
The previously failed rank3 cell is now independently accepted. The new
controller is launched with max_concurrent8/workers_per_gpu2; runtime admission
and actual process counts must confirm the handover before reporting it live.

Handover confirmed2026-10-15 12:42:30UTC: all four old worker PIDs were retained,
four new workers launched, and progress reports exactly two active sweep
processes on eachGPU0–3. Count21/280 accepted, no failures. Live free memory
before additional admissions was23,854/32,501/26,806/33,900MiB onGPU0/1/2/3;
each launch retained the8,192MiB minimum. No registration was restarted and no
unrelated process was stopped. The prior controller alone received SIGTERM.

## Preflight

### Continue-on-error audit and implementation

After the eight-process launch, rank2/cps8421/displacement/lambda0.01 exceeded
the unchanged FP32 transition max-error guard:0.0010928381234407425 voxel versus
0.001 allowed; RMS1.6594131178964186e-6 voxel. This is not CUDA OOM. The prior
controller intentionally stopped dispatch and drained to35 accepted, one failed,
244 blocked configurations. The old failed rank3 attempt remains preserved; its
approved attempt1 is accepted and must not be confused with an unresolved case.

User approved adding safeguards and resuming all unattempted configurations,
deferring scientific failures for review after the queue. Audit found parent
validation exceptions, malformed progress JSON, missing fast-exit PID metadata,
GPU telemetry failure, unresolved partial attempts on resume, nonpersistent OOM
budgets/reservations, historical-log OOM misclassification, unobserved hangs,
launch/storage errors, controller handover gaps and full-matrix-only reporting
could disrupt that policy. The new isolated controller/report adapter addresses
these without editing any hash-frozen numerical source or validation threshold.

New fault tests cover local failure followed by successful dispatch/worker/audit,
parent-validation exceptions, zero-exit missing markers, progress corruption,
partial/failed GPU telemetry, conservative persistent memory admission, durable
retry budgets, current-log-only OOM detection, fast child exit, process-spawn and
shared-journal failure, accepted-over-old-failed reconciliation, deferred restart,
pre-directory orphan adoption, watchdog identity/slot retention, and incomplete
report pairing/means. Execution requires both these tests and preserved-source
preflight. No new numerical smoke is needed: worker/auditor bytes are unchanged.

See execution-spec amendment for exact timeout/admission/failure policies.
No general protection against uninterruptible kernel/NFS hangs is claimed.
Queue/report artifacts use continue_on_error_v1; original progress before the
amendment is retained separately. Final reports keep time and memory diagnostics;
failed/unvalidated fields are excluded, not accepted with relaxed checks.

Verification:50 SVF numerical/recovery/report/scheduler tests passed, including
20 new continue-on-error fault tests. Scoped whitespace check passed. Proceed
with read-only numerical-source validation, accepted-run validation and full
artifact preservation inventory before production admission. No numerical
source file has been changed by this queue amendment.

Resume preflight confirmed35 complete,244 pending,one failed/deferred, and no
active old workers/controller. The new inventory records1425 preexisting files;
the old progress snapshot and all accepted/failed attempt artifacts are retained.
Eight new workers started2026-10-15 13:28:00UTC (06:28PDT), exactly two on each
GPU0–3, UID:GID1009:1020. Admission snapshot free memory was26,920/30,450/
30,878/38,004MiB onGPU0/1/2/3. At13:28:45UTC all eight were optimizing cps8
or cps4; no infrastructure errors or new failures. Status35 accepted,one
deferred,eight active,236 waiting. A new ETA requires observed audited throughput;
do not extrapolate startup-only status or treat the deferred cell as accepted.

Seventeen unittest cases passed, including existing P73 tests, all-stage and
both-penalty exact lambda1 value/gradient parity on identical CPU controls,
seven-weight loss/gradient scaling, rank/CLI validation, fresh zero-state
checkpoint selection and weighted histories. Synthetic full280-run reporting
passed:30 summaries,980 stages,840 integration rows,9800 label rows and
1400 paired rows, three figure formats, n/a DL resources and immutable
acceptance. Scoped whitespace/diff check passed.

Four native-size extreme-lambda smokes dispatched with source hashes frozen.
Measured memory from both worker and independent replay sets production
admission; process Torch memory and device-wide concurrent load remain separate.

Four smokes and four separate replay audits passed with no failures. Extreme
weights were verified in all four stage records. Worker peaks: velocity
allocated3.7257/reserved3.9941GiB; displacement allocated3.4032/reserved3.7266GiB.
Admission remains8GiB. Production controller launched on available GPUs0–3
with case-affine queues, all280 independent registrations and automatic final
reporting. Exact timestamps and active external workloads are in launch_events
and gpu_load artifacts. No existing process was stopped or reprioritized.
