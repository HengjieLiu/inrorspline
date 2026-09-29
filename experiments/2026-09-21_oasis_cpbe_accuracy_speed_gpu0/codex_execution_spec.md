# Accuracy-first CP-BE overnight first10 screen

## Authority, deadline and scope

User approved planning and execution sequentially on GPU 0 until 2026-09-21
09:00 America/Los_Angeles (PDT, UTC-07; 16:00 UTC). Interpret the colloquial
"PST" as local Pacific civil time and use the earlier cutoff conservatively.
No new work after 15:40 UTC, own GPU workers stop by 15:50 UTC, reserve the
remainder for a report. Never stop foreign processes. User removed the weekly
usage constraint: extensions require only completed main work and enough time.

This is exploratory tuning on seed-3 shuffled ranks 0–9, not held-out validation
or a publication-cohort replacement. CP-BE's prior FP32 equivalence failure
remains explicit. All candidates are compared against a fresh CP-BE fast arm,
not conflated with original dense BE. No paper edits or commits.

## Frozen protocol and candidates

OASIS organized split_v1, alpha1000, seed1, float32; six CPU threads; cuDNN
benchmark=False/deterministic=True; matmul TF32 false, cuDNN TF32 true, no
global deterministic-algorithm requirement. Pull normalized XYZ, full output
(160,224,192,3), exact dyadic prolongation, stage Adam states reset as before.
Use existing Docker image/mounts and UID/GID1009. Actual paths stay in private
configuration/external execution manifests.

Main candidates, completely specified by `overnight_candidates.py`:

1. Existing MR_A_fast_decay + cpbe, adaptive 400/800/1300 maxima.
2. Same, but last stage capped at50.
3. Fixed50/50/50.
4. Fixed50/50/100.
5. Fixed75/75/100.
6. Original adaptive geometry/windows/minimums, thresholds2x.
7. Thresholds5x, minimum50 each, window/check25, patience25.
8. Thresholds5x, minimum50 each, window/check10, patience20.
9. Fixed50/50/100 with stage-local cosine LR, floor20%.
10. Fixed50/50/100 with stage-local LR drops to50% and25% at50%/80% budget.
11. cps16/8/4/2, down8/4/2/1, LR.03/.03/.01/.003, fixed25/50/50/100.
12. Same four-stage geometry, adaptive maxima200/400/800/1300, minima30/40/50/60,
    patience20, window/check10, thresholds5x.

The normal 8/4/2 stage LRs remain .03/.01/.003. Cosine/step schedules change
LR before each Adam update; logged base and effective rates are both retained.
No optimizer/network/loss defaults change outside opt-in experiment actions.
The legacy method-key string contains8-4-2, but actual stage arrays and control
shapes explicitly define the four-stage variants; do not infer geometry from it.

After all main cohorts finish, try the fastest all-gate candidate with
cache+batch+cpbe, then AMSGrad, only if each full candidate can finish before
cutoff. Each extension retains all other selected settings. No labels enter
online optimization or early stopping. Two fresh-worker confirmation timing
blocks per selected winner/reference are last priority and deadline-gated.

## Runtime and evaluation

Fresh worker per candidate: rank0 cold, rank0 warm, ranks1–9 warm =11 timings.
No extra warmup before cold. Resume in a new worker requires an excluded full
candidate-specific warmup before any warm-labelled measurement. Timing is
case_load_complete→dvf_file_ready; all histories are exported after that endpoint.
Cold has one observation; warm SD is sample SD across cases. Record setup,
actual steps, stop reasons, peak allocated memory and requested/effective flags.

After each registration block, evaluate its ten warm fields in a separate
sequential GPU0 worker. Primary:35-label Dice and area-weighted HD95 at1mm,
existing missing-label rules, label counts and paired label differences.
Secondary: final full-resolution global NCC, full/foreground FD voxel BE and
diffusion, float64 Jacobians with crop2, positive-J natural-log population SD,
folding percentages and positive fractions. Foreground is raw fixed>0.
Tag FD BE separately from analytic training CP-BE. No metric time enters runtime.

Engineering screening gates versus the fresh reference (not clinical margins):
mean Dice drop<=.002, worst case<=.01; mean HD95 increase<=.10mm, worst case<=.50mm;
identical valid-label patterns; mean NCC drop<=.001; foreground folding increase
<=.50 percentage points; mean FG SDlogJ/FD BE/diffusion<=1.10 times reference.
Report primary gates separately from secondary guards and an accuracy/runtime
Pareto frontier; no opaque weighted composite. Keep all failures/outliers.
Final promotion needs independent data because first10 is being used for tuning.

## Verification, monitoring, provenance

Before formal work: syntax/CPU contracts, no-execution job manifest, Docker
image/mount/permission/GPU checks, pair/data hashes, every-candidate one-step
smoke and a separate metric smoke. Correctness-only main smokes may share one
worker; they are never headline cold/warm observations. Formal candidate workers
are still fresh. All132 main timing jobs and120 main quality
jobs are preregistered. A bounded run may finish fewer: mark incomplete honestly.

Continuously poll NVML GPU0 compute processes at~1Hz and job boundaries. Map
host/container worker PID using a random token and process start time. Keep
contaminated attempts; retry cold in a fresh process and warm after clearance.
Monitoring gaps>5s fail clean acceptance. Deadline/timeout signals target only
the verified current study worker, never other jobs or the container itself.

Freeze all relevant training/experiment sources before GPU smoke/formal work;
retain source snapshots, Git HEAD/dirty patch, dataset/environment manifests,
exact commands, requested/effective flags, all fields, attempts and monitor logs.
Incrementally refresh Markdown/CSV/JSON after every evaluated case. Generated
outputs remain under new external roots:

`${ARTIFACT_ROOT}/runs/oasis_cpbe_accuracy_speed_<timestamp>/`

`${ARTIFACT_ROOT}/repo_experiments/2026-09-21_oasis_cpbe_accuracy_speed_gpu0/<run-id>/`

## Reproduction and delivery

Use the host's standard-library-only controller from the repository root; all
dependency-sensitive data/registration/evaluation work runs inside Docker.
Private environment variables below identify the already-verified mounts; they
are not additional data copies. Choose a fresh run ID when sources change.

```bash
python3 -B -m scripts.experiments.mrdbscp842_acceleration.run_overnight \
  --container "$CONTAINER_NAME" --repo-container "$REPO_CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" --artifact-container "$ARTIFACT_CONTAINER" \
  --oasis-root "$OASIS_CONTAINER_ROOT" --run-id "$STUDY_RUN_ID" \
  --deadline 2026-09-21T09:00:00-07:00 --extensions --phase dry-run
# Repeat the same command with --phase all for automatic preflight + execution.
```

`scripts/reports/wait_oasis_cpbe_delivery.py` is the independent host completion
queue. It waits for `controller_EXIT.json`, then runs the CPU-only Docker
finalizer `scripts/reports/oasis_cpbe_overnight_delivery.py`. Exact invocation
and return codes are retained externally. The finalizer adds a four-panel
accuracy/speed figure, source/backend/timing/count checks, and checksums.

Primary review files: `report.md`, `accuracy_speed.png`, `results.json`,
`case_metrics.csv`, `paired_differences.csv`, `label_differences.csv`,
`timings.csv`, `evaluations.json`, and `delivery_validation.json`. Keep partial
status visible if interrupted. No failed or smoke measurements are presented
as a completed cohort; confirmation timings never silently inherit quality
from a different field.
