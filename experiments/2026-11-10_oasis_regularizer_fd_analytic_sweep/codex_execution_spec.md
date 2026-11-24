---
doc_type: codex_execution_spec
status: planned
---

# OASIS first10: FD and analytic regularizer sweeps

## Status, objective, and limitations

This is an experiment specification, not a launch record. Implementation,
verification, and production runs have not been performed for this experiment.

Compare diffusion, bending energy (BE), and Leow-style KL with C1 quadratic
continuation, each using finite-difference (FD) and analytic cubic B-spline
derivatives. Tune all six arms independently on rank0, then evaluate a frozen
nine-weight grid for every arm on the seed3 shuffled OASIS first10.

Known limitations:

- First10, including rank0, is exploratory test-set tuning, not held-out validation.
- KL-C1 alone does not penalize volume-preserving shear or guarantee invertibility.
- The C1 continuation is our engineering adaptation, not the original Leow solver
  or an exactly inverse-symmetric penalty over its entire extended domain.
- Canonical SD(logJ) excludes nonpositive determinants; folding and positive-J
  coverage must accompany it. Smaller SD(logJ) alone is not a success criterion.
- FD stencils and analytic voxel-center derivatives have different sampling and
  boundary supports. Preserve and report those definitions.
- Concurrent GPUs share CPU, storage, and other host resources. Report concurrency
  and contamination; these measurements do not establish isolated-system speed.
- Process-cold does not mean cold filesystem caches. One cold sample has no SD.
- cuDNN determinism does not guarantee bitwise deterministic grid_sample backward.

## Frozen data and backbone

- Dataset: OASIS/brain, ranks0-9 from
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Direction: first ID fixed, second ID moving; rank0 fixed0333/moving0115.
- Standard orientation, stored raw float32 intensities, no intensity rescaling.
- One absolute cubic B-spline displacement field, prolonged by exact crop-aligned
  dyadic knot insertion through cps8/4/2. No composition, SVF, cps1, or dense stage.
- Gaussian-antialiased pyramid factors4/2/1 and the existing endpoint-aligned
  continuous spline evaluation. Normalized-grid XYZ parameters, align_corners=True.
- Unmasked squared local NCC9, matching the existing DL-matched implementation;
  border image sampling and canonical nearest-neighbor/zero-padding label warps.
- Full image shape160x224x192; foreground evaluation uses fixed raw image>0.
- Fresh identity initialization for every registration; no sharing of optimized
  prefixes across weights, loss types, derivative backends, or cold/warm repeats.

| Stage cps | Pyramid factor | LR | Max updates | Min updates | Requested patience |
|---|---:|---:|---:|---:|---:|
| 8 | 4 | 0.03 | 400 | 40 | 20 |
| 4 | 2 | 0.01 | 800 | 80 | 35 |
| 2 | 1 | 0.003 | 1300 | 130 | 50 |

Reuse the existing median-window/check interval25, effective patience at least25,
relative total/data thresholds1e-4/5e-5, and stage-local stopping logic. Rates
remain fixed within each stage: the user selected existing stage rates rather
than within-stage LR reductions. Maximum-budget stops remain explicitly capped.

Reset Adam each stage: betas0.9/0.999, eps1e-8, no weight decay or AMSGrad. Select
the lowest stage total-loss state including initialization; save terminal state
and optimizer too. Use seed1, FP32, cuDNN benchmark=False/deterministic=True,
TF32 disabled. No label-based checkpoint selection or initial-loss normalization.

## Six objectives and derivative conventions

Each arm minimizes NCC9 + weight*R, with the same swept weight at all three stages.
Only its selected regularizer is active. KL-plus-diffusion is deferred.

| Arm | Regularizer |
|---|---|
| diffusion_fd | Existing unmasked forward-valid diffusion of total expanded field |
| diffusion_analytic | Exact cubic first derivatives of the total spline field |
| be_fd | Existing unmasked forward-valid pure/mixed second differences |
| be_analytic | Exact cubic pure/mixed second derivatives of total spline field |
| kl_c1_fd | Central-difference determinant of total expanded transform |
| kl_c1_analytic | Determinant from exact cubic first derivatives |

Use current evaluation-grid voxel units for displacement and derivatives.
Analytic derivatives require component scaling and the chain factors from native
coordinates to endpoint-aligned stage coordinates, not an integer downsample
shortcut. Preserve diffusion's component/axis mean and BE's component mean with
doubled mixed terms. FD diffusion/BE retain their existing valid-stencil means;
analytic diffusion/BE use voxel-center means over the full stage grid.

For both KL arms, evaluate J=det(I+Du) on the same two-voxel interior crop. Use
the total current transform, not det(Du), controls, or an incremental deformation.
The FD implementation must match the canonical central-difference Jacobian on
that support, while retaining autograd; the existing NumPy evaluator is not a
training loss. Analytic derivatives refer to the continuous spline itself.

`leow_kl_c1_spec.md` preserves the user-supplied definition. Fix epsilon=0.05 and
kappa=120.0; sweep the separate lambda_kl. Use raw (J-1)*log(J) for J>=epsilon
and the specified C1 quadratic below it. Guard the inactive logarithm branch,
retain negative-J gradients, and promote half precision if ever encountered.
FP64 validation must not be silently downcast. This experiment's KL-only protocol
supersedes the attachment's suggested smoothness-plus-KL experimental comparison.

The new interface must name regularizer, derivative backend, weight, epsilon,
and kappa. Serialize the active coefficient as lambda_diffusion, lambda_be, or
lambda_kl; inactive coefficients are zero. Kappa is never a regularizer weight.
Keep historical numerical sources/defaults unchanged; the new adapter must not
inherit misleading diffusion-only metadata for another loss.

## Coarse tuning, first10 grid, and historical reuse

For each arm independently, test rank0 weights1e-5/1e-4/1e-3/1e-2/1e-1/1/10/100/1000.
Use one shared zero-regularization control. A zero weight bypasses the regularizer
graph completely, so backend-specific code cannot change this control.

Each pilot setting is a fresh worker block: rank0 cold, followed by rank0 warm.
Use the warm rank0 Dice for tuning; retain cold metrics for QA, never choose the
better cold/warm outcome. No preliminary warmup or performance instrumentation
that creates a CUDA context in the measured worker before its cold registration.

Choose the best coarse power-of-ten weight by Dice, then symmetric ASD, HD95,
and smaller weight for exact ties. Require two lower and two higher positive
weights with Dice worse by more than1e-5. Extend the deficient side by decades
within1e-8 through1e6. Failed/nonfinite attempts do not count as worse results.
If the limits cannot bracket a peak, record flat_or_unbracketed and withhold that
arm's first10 expansion pending review. Report capped stages and numerical failures.

For a bracketed coarse winner10^k, freeze nine weights:

```text
10^(k-2), 3*10^(k-2), 10^(k-1), 3*10^(k-1),
10^k, 3*10^k, 10^(k+1), 3*10^(k+1), 10^(k+2)
```

For a winner1 this is0.01/0.03/0.1/0.3/1/3/10/30/100. Go directly to first10;
the user declined a separate refined-rank0 stage. Keep the grid independent for
each arm. Select first10 operating points by mean Dice, then symmetric ASD,
HD95, and smaller weight for exact ties. Report a cohort boundary winner as a
range limitation; do not silently expand or relabel the frozen primary grid.

The audit found70 completed matching-protocol first10 FD-diffusion cells at
0.01/0.1/0.3/1/3/10/100. The older analytic-BE sweep exists but used global NCC,
fixed LR0.01 and fixed iteration budgets, so it is historical context only.
No matched six-arm result matrix was found. Missing results must be generated
by real runs, never imputed or fabricated.

Historical fields may support provenance-validated reference/accuracy checks.
They cannot replace the fresh cold/warm performance blocks required below.
Main first10 accuracy uses the ten new warm fields from the corresponding block.
Do not mix old accuracy and new resource measurements as if from the same run.

## All-GPU scheduling and cold/warm resource protocol

Use every GPU visible to the configured runtime. The planning audit found four
matching GPUs, physical0-3. Rediscover IDs/UUIDs at launch; write their inventory
to private execution provenance, not hard-coded host paths in source.

- At most one experiment job per physical GPU. A pinned setting block occupies
  its GPU through worker exit; do not overlap registrations, smokes, or evaluators.
- Use four concurrent setting blocks on the current host whenever work is ready.
  Schedule deterministically across available slots and record assignment/order.
- Preserve unrelated jobs. Admit work only with sufficient free resources and
  no competing GPU compute process; never stop or repurpose another process.
- Use the configured Docker image/runtime and UID/GID from private configuration,
  not an inherited numeric user from another host. Repo/data read-only, artifacts RW.
- Fix six intra-op CPU threads and one inter-op/BLAS thread per worker. Record
  affinity, active workers, backend flags, versions, and GPU UUID with every result.
- Admission requires max(8GiB, measured smoke peak reserved+2GiB) free GPU memory.
  Project remaining storage from measured retained artifacts plus30% margin.
  Pause admissions rather than silently lowering fidelity when resources are short.

Every final arm/weight uses one new persistent worker on one GPU, strictly:

```text
rank0 cold -> rank0 warm -> rank1 warm -> ... -> rank9 warm
```

The ten warm cases start independently from identity with fresh optimizer state;
only process/CUDA/library/allocator state persists. Do not empty the CUDA cache
between them. No metric evaluation or extra registration intervenes in this block.
The shared zero-weight first10 control gets the same11-run block once.

For six successfully bracketed arms, the first10 matrix contains54 positive
settings plus one shared zero control:55 blocks and605 timed registrations,
comprising55 cold and550 warm rows. Primary positive-weight accuracy remains
540 case cells, plus10 zero-control cells; cold rank0 repeats are QA/resource rows.
Initial coarse pilots add55 two-run blocks,110 registrations before extensions.
Thus the nominal full protocol is715 registrations before smoke, extension, or
retry work. Pilot and cohort blocks remain separate even when a setting overlaps:
do not combine warm cohorts across worker processes or retain idle pilot workers.

Measure with monotonic clocks and CUDA synchronization at timing boundaries:

- `registration_sec`: case_load_complete through dvf_file_ready, including stage
  preparation/optimization, final dense expansion, CPU conversion and NPY writing.
- `worker_startup_sec`: parent launch through ready state, including imports and
  device initialization; reported separately for each fresh worker.
- `case_load_sec`, per-stage preparation and optimizer-loop durations,
  final-DVF preparation/write duration, and case-start-to-DVF wall time.
- Cold launch-to-first-DVF wall time, so CUDA/import startup is visible even
  though the historical registration endpoint excludes it.
- Evaluation/reporting/queue wait durations separately, never added to headline
  registration time. Do not sum nested or overlapping intervals.

Track memory per case and stage:

- Reset PyTorch peak counters at case_load_complete and record allocated/reserved
  baselines, peak allocated/reserved, and end values through dvf_file_ready.
- Capture stage peaks separately without losing the case-wide maximum; keep
  explicit maxima across stage counter resets, including stage setup/finalization.
- Warm reserved memory includes retained allocator cache. Report its baseline
  and peak; do not present a reserved delta as total memory demand.
- Use a CPU-only monitor sampling exact worker PIDs/descendants at200ms plus
  phase boundaries. Retain per-process NVML usage, whole-device usage, free memory,
  other compute PIDs and timestamps. NVML peaks are sampled lower bounds;
  PyTorch allocated/reserved peaks are distinct measures, not interchangeable.
- Sample worker/descendant host RAM RSS on the same clock and report per-case
  peak RSS, explicitly distinguishing host memory from GPU memory.
- Record setup/load and evaluation memory separately. Monitoring must not create
  a CUDA context in the controller or run deep profiling inside headline runs.

Validate exact worker identity using PID and start time. Flag overlapping GPU
compute, monitor failure, interrupted sequences, and missing telemetry. Preserve
all attempts; rerun an invalid timing block in a fresh worker after clearance.
Never stitch its cold/warm sequence across worker identities. Finite registration
outputs may remain diagnostic but cannot fill accepted performance rows silently.

## Metrics, reports, and artifacts

Separate registration and evaluation phases within each coarse/extension/final
wave: finish its timed registration blocks before running its saved-field
evaluators, using all available GPUs with the same one-job-per-GPU slot limit.
Keep cold/warm-rank0 QA separate from the ten-case warm cohort. This prevents
untimed metric jobs from competing with timed registrations on the shared host.

For every case retain all35-label Dice/HD95/area-weighted symmetric ASD rows,
missing-label reasons and counts; canonical full/foreground Jacobian statistics;
FD and analytic BE/diffusion in existing voxel/normalized variants; folding and
NDV. Add both-backend KL-C1, min/p01/median/p99 J, J<=0 and J<epsilon fractions.
Preserve raw determinants and existing metric definitions. Canonical SD(logJ)
uses natural log of positive J, crop2, and population SD within each volume.
Across-case report SD uses ddof1. Undefined metrics retain explicit reasons and
support counts; never drop cases or clamp determinants to conceal failures.

Every report must contain:

- Six-arm accuracy/regularity tables and weight curves, per-case/per-label rows,
  rank0 versus ranks1-9 summaries, and accuracy-versus-regularity plots for all
  three accuracy metrics, especially foreground SD(logJ) beside folding/coverage.
- One resource row per setting and phase: GPU/worker provenance; cold registration,
  startup and launch-to-DVF times; warm registration mean/sample SD, median and
  range; per-stage times/updates/stopping; and separately timed evaluation.
- The paired rank0 cold-minus-warm0 time/ratio. Cold0 versus the ten-case warm
  average is cohort-dependent and must not be labeled a pure cold-start overhead.
- Cold allocated/reserved/NVML peaks; warm mean/sample SD and maximum for each
  memory measure; warm reserved baselines; case-level rows and missing/contaminated
  counts; and sampled host RSS peaks. Pilot warm0 has n=1, hence no warm sample SD.
- Completed/expected counts for blocks,605 first10 timing rows and550 warm accuracy
  rows, with positive versus zero controls separated. Partial cohorts remain partial.
- Primary new measurements distinct from historical references; no invented zero
  values for historical timing/memory that were never collected.

Write CSV/JSON alongside the readable Markdown report and plots. Keep all
generated artifacts under `${ARTIFACT_ROOT}/runs/<new-run-id>/` and
`${ARTIFACT_ROOT}/repo_experiments/<experiment-key>/<new-run-id>/`, with source
snapshots/hashes, dirty patches, inputs/split hashes, commands, run manifests,
all attempts, fields, best/last checkpoints, optimizer states and monitor logs.
Never overwrite historical artifacts. Keep actual host/storage mappings private.

## Implementation and verification gates

Add an isolated `scripts/experiments/oasis_regularizer_sweep/` adapter, focused
tests and an implementation note; update the registry before a real run. Do not
modify historical numerical sources or unrelated dirty-tree work.

Before production verify:

1. KL reference values, C1 value/gradient matching, active negative-J gradients,
   float64 gradcheck, mask normalization and expected-range dtype behavior.
2. Identity, translation, affine scaling and shear; determinant sign/axis/units;
   analytic derivative numerical checks and end-to-end control-point gradients.
3. All pyramid scaling/derivative orders, knot insertion, canonical FD-J parity,
   and new FD-diffusion loss/gradient/optimizer compatibility with the old path.
4. Weight0 bypass, state-indexed histories, label-free checkpoint selection,
   backend metadata, reuse rejection, bracketing and complete-cohort aggregation.
5. Fresh-worker cold status, cold/warm ordering, identity/optimizer resets, no
   evaluation within timed blocks, CUDA synchronization, memory-counter windows,
   sampled-monitor accounting, GPU slot exclusion and interrupted-block handling.
6. Six full-geometry one-update smokes before coarse pilots and extreme-weight
   smoke checks before range extensions. Smokes never warm a measured worker.
7. Independently replay saved checkpoints/fields/losses and recompute report
   statistics; require the frozen block/case counts for full acceptance.

This file authorizes no staging, commit, sync, deletion, or paper-archive change.
