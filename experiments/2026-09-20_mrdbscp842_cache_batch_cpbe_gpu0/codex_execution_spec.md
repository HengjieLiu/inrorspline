# P2ad: first10 cache + batch + control-space BE

Status: completed and acceptance-checked; directory date is UTC.

## Frozen experiment

Physical GPU 0 only, serial Docker workers with UID/GID 1009:1009. Resolve the
existing container, read-only OASIS mount, and writable artifact mount from private
configuration. Preserve seed3 shuffled ranks0-9, MR_A_fast_decay, alpha1000,
seed1, float32, cuDNN benchmark=False/deterministic=True, matmul TF32=False,
cuDNN TF32=True, deterministic_algorithms=False, CPU threads6/1.

Arms, in order: reference; cpbe; combo_cache_batch_cpbe. The last arm enables
only cache,batch,cpbe. No NCC caching, loss buffering, optimizer or stopping change.
Full cps8/4/2, downsample4/2/1, maximum400/800/1300, LR.03/.01/.003,
minimum40/80/130, patience20/35/50, check25, relative total/data1e-4/5e-5.

Each arm receives a fresh worker: rank0 cold, repeated rank0 warm, ranks1-9 warm.
No preliminary warmup. First run 33 endpoint-only jobs; then 33 separate component
jobs in the same order. Retain all fields and separate repeated-rank0 paths.
Headline endpoint is case_load_complete through dvf_file_ready. CUDA setup and
case loading are separate; metric evaluation and post-DVF packaging are excluded.
Cold does not imply cold filesystem caches. One cold sample has no SD.

## Components

Opt-in nested host and CUDA-stream intervals: stage pyramid/identity/optimizer
setup; control allocation; exact prolongation; full forward; spline kernel
creation/cache access and each axis/crop/layout; coordinate addition; grid_sample;
NCC; BE and six derivative terms or CP differences/QR factors/transfers/axis
projections/reduction; loss assembly; zero_grad/backward/Adam; scalar transfers,
logging and stopping; stage statistics; final field expansion/CPU copy/NPY write.
Resolve CUDA events at stage boundaries. First iteration and remaining iterations
are separated. Inclusive forward/BE/expansion parents are not added to children.
Host dispatch/wait and CUDA-stream spans overlap; spans are not kernel-only time.
Instrumentation costs and uncovered host time remain explicit. Report components
for cold, warm mean/sample SD, each stage and whole run; retain per-case raw data.

## Evaluation and acceptance

Evaluate 33 headline outputs in a later fixed-backend GPU-0 process. Aggregate
the ten warm cases per arm; retain cold-rank0 QA. Dice and HD95 use the existing
35-label missing-label rule, nearest label warp, 1mm surface spacing. SDlogJ uses
jacobian_metrics_voxel: voxel ZYX displacement, float64 central derivatives,
two-voxel crop, natural log over J>0, population SD, raw fixed image>0 foreground.
Report full/foreground SDlogJ, folding and positive fractions, paired differences
against reference and CP-BE, label outliers, field errors, steps and stop categories.

Validate all 66 fields float32/finite/normalized XYZ/(160,224,192,3), backend flags,
unique IDs, job sequence, per-worker provenance, source freeze and separate metrics.
Diagnostic/headline parity uses vector RMS<=.005 voxel/max<=.05 and equal steps/
stop categories; mismatches are flagged, never substituted into headline timings.
BE non-equivalence relative to original is expected evidence, not a run blocker.

Run syntax/CPU contracts, manifest dry run, six one-step smokes (3 arms x2 modes),
and GPU field/phase/cache accounting checks before formal jobs. Monitor exact
worker host PID plus descendants, sample NVML ~1s and at boundaries; query errors
invalidate attempts. Retry contaminated warm jobs after clearance and cold jobs
in a new worker. Preserve all attempts and never stop unrelated processes.

## Replay and storage

Host entrypoint: python3 -B -m
scripts.experiments.mrdbscp842_acceleration.run_be_combo_first10, with explicit
--container, --repo-container, --artifact-host, --artifact-container, --oasis-root,
--gpu 0, --run-id mrdbscp842_be_combo_<UTC timestamp>, and --phase dry-run/preflight/all/report.
Use all only after inspecting smoke validation. Resume only with frozen sources.
Runs: ${ARTIFACT_ROOT}/runs/<run-id>; reviews:
${ARTIFACT_ROOT}/repo_experiments/2026-09-20_mrdbscp842_cache_batch_cpbe_gpu0/<run-id>.
Retain commands, code hashes/copies, dirty patch, Git/image/data/backend metadata,
monitor logs, fields, component/metric CSV/JSON, acceptance and Markdown report.
No paper changes, no commit, and no modifications to P2ac generated evidence.

### Long evaluation recovery

Following the accepted run, the dedicated workflow gained
`--evaluation-timeout-seconds` (default 14400). This changes only the untimed
evaluation watchdog; historical commands retain their one-hour default. The
accepted measurements predate this host-only amendment and retain their original
source snapshots. New full replays should use a new run ID with the current
dedicated runner, without needing adoption.

The original historical worker controller had a one-hour command watchdog,
including the entire untimed evaluation command. If adopting that older running
version because the deadline is shorter than the existing CPU
Jacobian/HD95 pass, adopt the still-running evaluator before the deadline using
`python3 -B -m scripts.experiments.resume_mrdbscp842_be_combo_evaluation`.
Pass the same explicit connection/run arguments (without `--gpu` or `--phase`),
plus `--worker-name`, `--controller-pid`, and `--controller-start-ticks` from the
verified live process identity. Never guess a PID. The helper checks the exact
33-field command, completed 33+33 registrations, frozen sources and both process
identities before signalling only the controller; the GPU evaluator is retained.
Its default `--timeout-seconds 14400` is an untimed orchestration deadline, not a
change to registration or metric semantics. Recovery source/commands and an
overlapping monitor are captured externally. Use only once for a run: an existing
recovery directory or accepted evaluation is a fail-closed guard.
