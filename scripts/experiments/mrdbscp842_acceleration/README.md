# MR-D-BSCP 8-4-2 acceleration experiments

## Accuracy-first overnight screen (P2af)

`run_overnight.py` runs12 fixed/adaptive/LR/16-8-4-2 candidates on the seed3
first10, with cold+10warm timing and separate10-case Dice/HD95/NCC/regularity
per arm. `overnight_candidates.py` freezes schedules and engineering gates;
`overnight_actions.py` supplies scoped LR/AMSGrad contexts and untimed metrics;
`overnight_report.py` refreshes Markdown and CSV/JSON after each evaluated case.
Pass the common path arguments below, a NEW `oasis_cpbe_accuracy_speed_<timestamp>`
run ID, and an explicit timezone-aware `--deadline`. Use `--phase dry-run`, then
`--phase preflight`, then `--phase all`; the latter resumes accepted work.
`--extensions` enables cache/batch and AMSGrad follow-ups only AFTER all main
cohorts finish and the deadline budget permits. No weekly-quota condition remains.

The controller polls GPU0 compute processes at~1Hz, retries contaminated work,
and signals only its identity-verified worker at the deadline. It stops new work
20min before the deadline and reserves the final10min without GPU work for
reporting. `state.json`, `primary_rows.json`, `quality_rows.json` and the external
review `report.md` expose progress and partial-cohort status. First10 selection
is exploratory; it cannot establish held-out registration equivalence.

## Three-arm BE combination follow-up (P2ad)

`run_be_combo_first10.py` is the dedicated follow-up: original fast, CP-BE alone,
cache+batch+CP-BE. Each arm has rank0 cold followed immediately by warm ranks0-9
in one fresh worker. All 33 endpoint-only jobs precede all 33 separate detailed
jobs, and metric evaluation follows registration. Historical commands/defaults
below remain unchanged. Use a NEW `mrdbscp842_be_combo_<timestamp>` run ID.

Pass the same explicit container/repo/artifact/OASIS/GPU arguments documented
below to `python3 -B -m scripts.experiments.mrdbscp842_acceleration.run_be_combo_first10`.
Run `--phase dry-run`, then `--phase preflight`; inspect `smoke_validation.json`
before `--phase all`. The `all` phase resumes accepted blocks with frozen source
hashes, evaluates Dice/HD95/SDlogJ, reruns source contracts, and writes the final
acceptance/report. `--phase report` regenerates outputs without GPU registration.

The dedicated first10 workflow uses `--evaluation-timeout-seconds 14400` by
default for the long, untimed HD95/SDlogJ pass; the historical runner's one-hour
default remains unchanged. For adoption of an already-running older controller,
the execution specification documents
`scripts/experiments/resume_mrdbscp842_be_combo_evaluation.py`: it adopts the
verified existing evaluator with overlapping monitoring and a longer deadline,
without restarting registration/evaluation or changing frozen sources. Recovery
identities, commands and monitor evidence remain outside Git with the run.

`components.py` is an opt-in diagnostic context enabled by `--timing-detail components`.
It reproduces original expansion/CP-BE operations and records nested host/CUDA
spans. These diagnostic runs are NEVER headline timing or memory observations.
Parent/child spans and host/GPU clocks overlap; CUDA spans can include stream-idle
gaps. Cold-only basis construction remains visible instead of being warmed away.
`test_components.py` checks arithmetic/gradient/Adam parity, nesting, cache counts,
and SDlogJ conventions. `be_combo_report.py` exports full component CSV/JSON,
case/label metrics, paired field/step differences, and a readable detailed report.

See `experiments/2026-09-20_mrdbscp842_cache_batch_cpbe_gpu0/` for the frozen
protocol and known limitations. All follow-up artifacts use separate external
roots; the completed P2ac evidence remains unchanged.

Opt-in implementations and a reproducible first-round experiment. Production
defaults remain unchanged. See the source-only execution specification in
`experiments/2026-09-19_mrdbscp842_acceleration_gpu0/` for scientific gates and
the technical log for limitations and user-approved amendments.

## Components

- `backends.py`: isolated kernel caching, grouped-to-batched expansion, fixed
  NCC statistics, buffered loss rows, and sampled control-space BE.
- `checks.py`: CPU/FP64 oracle and GPU/FP32 value/gradient/Adam checks.
- `test_contracts.py`: runner defaults, module activation, timing, stopping,
  nested-grid shapes, loss-history and monitor regression checks.
- `run_suite.py`: host-only orchestration, source snapshots, sequential Docker
  workers, exact host-PID identification, retry and provenance records.
- `nvml_monitor.py`: read-only GPU compute-process sampling, no CUDA context.
- `worker.py`: registration, operator benchmarks, diagnostics and separate
  Dice/Jacobian/HD95 evaluation in Docker.
- `report.py`: deterministic report/tables/decision regeneration from artifacts.
- `validate.py`: final job/evaluation counts, timing endpoint and backend checks,
  field validity, source/PID provenance, and retained BE-warning acceptance audit.
- `field_regions.py`: CPU-only foreground/background localization of full-DVF
  differences using raw fixed-image-positive foreground, without relaxing gates.
- `final_checks.py`: retained syntax, fourteen CPU contracts, diff and tracked-notebook
  checks, without initializing CUDA.
- `continue_study.py`: optional completion queue after an existing controller;
  serializes reporting, selection, checks, confirmation, evaluation and acceptance.

## Local setup and replay

Resolve private paths/container settings from ignored local configuration. The
container must see the repository and read-only OASIS split, and have a writable
artifact bind. Use the recorded image/runtime and UID/GID 1009:1009. GPU 0 is
the only permitted physical device for this experiment; workers mask it to CUDA 0.

Set `CONTAINER`, `REPO_CONTAINER`, `ARTIFACT_ROOT` (host), `ARTIFACT_CONTAINER`,
`OASIS_ROOT` (container) and a new `RUN_ID` outside the repository. On the host:

```bash
python3 -B -m scripts.experiments.mrdbscp842_acceleration.run_suite \
  --container "$CONTAINER" --repo-container "$REPO_CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" --artifact-container "$ARTIFACT_CONTAINER" \
  --oasis-root "$OASIS_ROOT" --gpu 0 --run-id "$RUN_ID" --dry-run
```

Repeat the command without `--dry-run`, with `--phase preflight`, then
`--phase baseline`. Do not run phases concurrently. Inspect A/A quality and
operator checks. Run `--phase pilot` and `--phase characterize`; the latter is
the explicitly authorized first10 BE quality characterization. Use
`--phase evaluate` to run metrics in a separate sequential GPU-0 worker and
rebuild reports. `--phase report` only rebuilds the report (no registration).

After examining pilot evidence, `--phase select` writes the selection manifest
and its evidence-backed rationale to the external run root. It combines only
isolated non-BE variants passing every preregistered gate in both modes; it does
not silently waive a failed gate. Inspect that manifest, rerun `--phase snapchecks`
to include the combination, and run `--phase bridge` for the instrumentation
comparison. Then `--phase confirm` runs
five paired first10 repetitions in both fixed-reference-step and adaptive modes,
plus three fresh-worker cold repeats per arm. Evaluate again afterward.
If no candidate qualifies, inspect the negative result rather than launching an
empty or fabricated combination.

For an already-running pilot/evaluation sequence, the same connection arguments
can be passed to `python3 -B -m
scripts.experiments.mrdbscp842_acceleration.continue_study --after-pid CONTROLLER_PID`.
Use the PID of the whole predecessor shell chain, not only its current child.
The queue verifies the run ID, records the PID start time, waits for that exact
process to exit, and stops on any failed phase. It writes
`completion_queue.json` and per-phase logs outside Git. Do not start two queues.
The confirmation entry guard requires passing isolated operator checks and all
15 saved-control gradient checks for the selected feature set.

Cold attempts require a fresh worker. Warm retries retain their warmed worker;
resuming a partially finished block creates a new worker and an excluded warmup.
Every attempt remains in its unique directory. Do not delete an interrupted
worker or its artifacts indiscriminately: inspect its exact PID/token and mailbox
before a targeted cleanup. Reusing a run ID is only for resuming the same protocol,
not for silently mixing new implementation variants.

## BE interpretation

The prototype transforms finite control differences with reduced QR factors of
the sampled lower-degree basis on the exact finite crop. The resulting energy is
a sum of squares with the original component/voxel mean, not an integrated BE or
an unregularized quadratic-form subtraction. No epsilon jitter or loss clamp is
introduced. Factor construction is cached by geometry, dtype and device; first
construction costs are included in cold timing.

Strict FP32 trajectory tests can fail near nullspaces even when FP64 algebra
matches. The user authorized registration tests despite this warning. Report
numerical equivalence and empirical registration quality separately. HD95 uses
the existing 35-label, area-weighted surface-distance backend at 1 mm isotropic
spacing, with absent-label NaNs; no new HD95 noninferiority margin is implied.

## Outputs

Only sources/specifications are in Git. External roots retain source hashes and
copies, dirty patches, exact worker commands, numerical checks, monitoring logs,
all attempts, final float32 normalized XYZ fields, profiling/diagnostic files,
timing CSVs, machine-readable decisions and the readable Markdown report.
`report.py` does not launch CUDA work. Identical-DVF evaluation reuse is explicit
and keyed by SHA-256 plus case and metric set.
