# P79: First100 B-spline842 displacement / bounded2 / refined-SVF sweep

## Approval and scope

Approved2026-10-16. Complete the seed3 shuffled ranks0–99 comparison at
lambda0.01/0.1/0.3/1/3/10/100: 2100 cells, 570 accepted read-only reuses and
1530 fresh registrations. No dense/cps1/symmetric arm and no new lambda search.

- Displacement: reuse430 (100 each at.1/.3/1/3,10 each at.01/10/100); run270.
- Bounded2 maps/level: reuse70 accepted P76; run630.
- Refined single total SVF with velocity diffusion: reuse70 P74; run630.
- The reused displacement outputs are separately saved/scored P43/P48 cps2
  prefixes, not the final dense-refinement outputs. No prefix warm starts for
  missing registrations, and no reuse of different fixed-stage-lambda sweeps.

## Frozen numerical definitions

All levels use the same swept lambda, raw squared local NCC9 and forward-valid
unmasked component-/axis-mean diffusion in current loss-grid voxel units.
Pyramid factors4/2/1, normalized LR.03/.01/.003, max400/800/1300,
min40/80/130, requested patience20/35/50, median/check25, effective patience
at least25, relative total/data thresholds1e-4/5e-5. No LR reductions.
Fresh Adam every stage/map; lowest total-loss state including initialization.
Seed1, FP32, cuDNN benchmarkFalse/deterministicTrue, TF32off; CUDA sampling
backward is not guaranteed bitwise deterministic.

Displacement refines one prolonged cubic field and regularizes total
displacement. Bounded2 retains six independently initialized increments with
validated0.99 tanh bounds, previous-o-current pull composition and two full
Adam schedules per level; regularizes total composed displacement. Refined
SVF prolongs one velocity, native seven-step exponentiation, velocity diffusion;
eight/ten-step reintegrations and inverse diagnostics remain auxiliary.
Historical objective/optimizer source files stay unchanged. Process-local
adapters only widen rank guards, naming, manifests and scheduling.

## Execution and verification

Use UID:GID1009:1020, GPUs0–3, at most two workers/auditors per GPU and eight
total. Require8GiB or measured smoke peak+2GiB, whichever larger, per slot;
preserve reservations through exit and pin independent audits to original GPU.
Require800GiB initial disk capacity,100GiB runtime guard,32GiB host free RAM.
Do not modify existing jobs. Preserve failures; continue other configurations;
one OOM retry at queue tail. Shared source/storage problems pause admissions.
Durable pre-spawn intent, PID/start-time reconciliation, lock, telemetry and
watchdog inherited from the reviewed P74 queue. Preserve all prior attempts.

Revalidate570 reusable field/model/history hashes, finite geometry, complete
markers, case/input/split identity, frozen protocol and canonical metric rows.
Store stat-bound hash receipts externally; reference fields remain read-only.
Six rank10 native-volume smokes (three families x lambda.01/100), one update
per stage/map, native loss parity, serialization and independent replay gate
production. All fresh runs need independent checkpoint/loss/history/metric
replay. Bounded checks and SVF FP64-gated transition checks remain unchanged.

Approved metadata-only recovery uses execution `metadata_recovery_v2` beneath
the same artifact family. The original failed preflight remains read-only.
Worker settings contain only four validated data/code paths, not controller
lists/flags. Revalidate and copy the570 old hash receipts under unchanged
scientific contracts/numerical sources; rerun all six smokes with the new source
manifest. No mathematical, optimizer, tolerance, cohort or lambda changes.

## Reporting

External run/review family:
`oasis_seed3_first100_bspline842_displacement_bounded2_svf_velocity_ncc9_lambda_sweep_v1`.
CSV/JSON/Markdown per-case, per-label and stage diagnostics; complete-only
mean/sample-SD groups for first10/additional90/first100. Five headline metrics:
Dice,HD95,symmetricASD,positive-J fixed-raw-FG SDlogJ,FG J<=0 percentage.
Original100-case VFA/SITReg references, paired differences,2x3 ARC means above
sample-SD ellipses. Acceptance only after2100 complete validated cells and
CSV statistic readback. Partial reports explicitly incomplete, no accepted
full-cohort marker. Keep runtime/memory split by historical/new provenance;
unknown historical memory is missing, not zero.

## Limitations

Approved scoped numerical continuation: ranks29/44,842/velocity/lambda.01 only.
Rank29 retains its selected cps8 checkpoint; rank44 retains selected cps8/cps4.
Resume with fresh Adam only at the unfinished levels. Policyv3 requires exact
prescribed FP32 knot insertion, unchanged velocity allclose and FP32 field
max<=.002/RMS<=1e-5 voxel. Failed original field allclose additionally requires
ideal FP64 velocity/field max<=1e-9/1e-8 and actual-rounded-control FP64 field
max<=2e-4/RMS<=1e-6. No field, optimizer, precision or seven-step integration
change. Use separate transition_continuation_v3 code/artifacts. Two full-size
smokes run one update at every unfinished stage then independent replay; only
after both pass run the two continuations, pinned to their originalGPU3/2.
Fresh memory admission8GiB or smoke peak+2; no changes to other GPU jobs.
New reports retain2098 results and explicitly separate continued-worker time
from retained-prefix optimizer time. Historical report releases remain intact.

First10 informed tuning and additional90 were previously inspected; no untouched
confirmatory-test claim. Bounded2 receives twice the optimizer budget. Equal
velocity/displacement lambda is not equal-strength regularization. Coarse-grid
field evaluation follows each accepted representation. Shared workload timing
is diagnostic. Numerical folding remains measured, never silently rejected.
