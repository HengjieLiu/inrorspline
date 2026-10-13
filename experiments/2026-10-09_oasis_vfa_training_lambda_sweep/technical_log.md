# VFA training sweep technical log

## Known limitations

Historical benchmark-mode CUDA is not bitwise deterministic. Source provenance
is hash-frozen rather than an immutable April2025 source snapshot. GPU0 has
38.6GiB free versus the prior40.1GiB GPU-wide VFA peak. User explicitly directs
trying lambda10 there without waiting; on OOM preserve failure and append one
retry toGPU3 after its lambda3 train/evaluation. No other workload is modified.

## Approved execution and implementation

Final assignment:GPU0=lambda10, GPU1=0.1, GPU2=1/3, GPU3=3. Supersedes the
earlier mapping discussed in chat. Each slot waits for its P49 SITReg evaluation
to finish, then native200-epoch VFA training and automatic detailed100-case
evaluation. GPU0 alone has the explicit tight-memory override.

Separate VFA helper/runner/evaluator/controller added. Preserve all existing
SITReg and common metric sources. The VFA runner keeps TWO sequential Adam
updates per pair and forward-only validation, not SITReg joint-loss semantics.
Evaluator imports identical P49 detailed-label functions.

## Preflight passed and production launched

2026-10-09 23:18 PDT: detached controller PID 66434 launched in the artifact-store
container as UID:GID 1009:1020. GPU0 lambda10 PID 66437, GPU2 lambda1/3 PID 66440,
and GPU3 lambda3 PID 66447 started; GPU1 lambda0.1 waits for its P49 evaluation.
GPU0's launch explicitly includes `--allow-tight-memory`; all other scientific
settings remain unchanged. The controller automatically schedules first100
evaluation after each training/first10 process completes successfully.

Preflight evidence in `oasis_vfa_ncc9_diffusion_training_preflight_v1/`:

- Nine VFA CPU tests passed, including exact native sequential-update semantics,
  final assignment, dependency gating, memory override and OOM fallback policy.
- Historical rank0 field reproduced bitwise (maximum and RMS voxel error zero).
  Canonical Dice/HD95/ASD: 0.79682034087069 / 2.0485631555197186 /
  0.4400145870843727. Exact field serialization and 35 per-label rows verified.
- Native full30 validation Dice 0.8115036169543193 versus checkpoint
  0.8115048678362494 (absolute difference 0.0000012508819301; tolerance 0.0001).
- All four fresh lambda pair smokes passed: two sequential Adam updates each,
  finite gradients, and exact checkpoint roundtrips. All runs require the same
  seed42 initial-state hash and preflight source hashes.
- Free memory at production launch: GPU0 39513 MiB, GPU2 44330 MiB,
  GPU3 48515 MiB. GPU0's conservative 42 GiB gate was bypassed as requested.

Run root: `${ARTIFACT_ROOT}/runs/oasis_vfa_ncc9_diffusion_training_lambda_sweep_v1/`.
Inspect `queue.json`, candidate `progress.json`, and preserved attempt logs for
live status; this launch record is not a claim of completed training.

## GPU0 OOM and automatic fallback

At 23:19:58 PDT the controller recorded GPU0 lambda10 attempt01 exit code 1
with CUDA OOM during the first pair's backward pass (5.54 GiB allocation request;
1.03 GiB device memory free). The failed attempt, manifest and full traceback
remain under `vfa_lambda_10_seed42_attempt01/`; no failed outputs were removed.
The controller set attempt02 to `waiting_for_gpu3_retry`, next stage `train`:
fresh seed42 training will start after GPU3 lambda3 finishes training and its
first100 evaluation. There is no additional retry beyond this approved one.

At the 23:20 PDT check, GPU2 lambda1/3 had completed 20 training pairs and GPU3
lambda3 27 pairs, with finite losses. Their initialization hashes exactly match
the preflight. GPU1 lambda0.1 remains queued while its SITReg evaluation is at
53/100 cases. All existing workloads remain untouched.

## Completed sweep and combined ARC export

The queue now reports all four VFA training/first100 evaluation jobs complete;
lambda10 uses the approved successful GPU3 attempt02, with failed GPU0
attempt01 preserved. The user-requested four-method ARC figure includes all
four new VFA weights and the unchanged historical lambda1 point. Each100-case
cohort and its recomputed mean/sample SD were validated before plotting.
See `experiments/2026-10-10_oasis_fedda_training_lambda_sweep/technical_log.md`
for the combined19-point/1900-case2×3 export and its interpretation.
