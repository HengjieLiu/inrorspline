# Technical log

## Known limitations and current status

User authorized implementation and launch on2026-11-10. Implementation and CPU
verification, full-volume preflight and production are complete. The experiment
ran2026-11-10T09:18:11Z through13:25:20Z (01:18-05:25PST), using GPUs0-3 with
one worker/GPU. All55 first10 groups passed registration/metric acceptance.
The first10 is exploratory; positive-J-only SD(logJ), KL shear insensitivity,
finite-penalty folding behavior, derivative-support differences, concurrent-host
timing and process-cold semantics are explicit limitations in the execution spec.

## 2026-11-10: protocol decisions and read-only audit

- Selected local squared NCC9 and the existing prolonged MR-D-BSCP8/4/2 pyramid.
- Selected fixed within-stage LR0.03/0.01/0.003 with existing adaptive stopping,
  rather than adding within-stage LR reductions.
- Expanded the initial KL-only comparison to FD and analytic variants for all
  three regularizers: diffusion, BE, and KL-C1. No combined smoothness-plus-KL arm.
- Each arm gets independent coarse rank0 tuning; require two worse weights on
  each side before the nine-weight first10 sweep. No separate refined-rank0 stage.
- Found70 completed NCC9/FD-diffusion first10 outputs on the seven historical
  weights and legacy analytic-BE outputs under a mismatched global-NCC protocol.
  This was a presence/protocol audit, not full hash/replay acceptance.
- The runtime audit exposed four matching GPUs0-3. User requested all GPUs,
  one job on each GPU, and per-setting cold/warm memory and time in reports.
- Adopted fresh setting workers, rank0 cold then ranks0-9 warm for final cohorts;
  pilot settings use rank0 cold then rank0 warm. Parameters always restart from
  identity; process state alone is warm. Preserve separate timing/memory windows.
- Performance requirements supersede registration reuse for the final matrix:
  every setting needs fresh timing blocks. Main accuracy comes from their warm
  fields; old fields remain distinct validated references when available.
- Nominal counts before extensions/retries:110 pilot registrations and605 final
  registrations,715 total. Final resource rows cover54 positive settings and one
  shared zero control; final primary accuracy has540 positive cells plus10 controls.

## Supplied loss specification

`leow_kl_c1_spec.md` preserves the supplied attachment verbatim. Its linked
reference Python files and figures were not attached. Statements about standalone
test results in that document are supplied provenance, not tests performed here.
The experiment's KL-only objective and six-arm schedule are defined by
`codex_execution_spec.md`, including its explicit override of the attachment's
suggested combined-regularization experiment.

## Verification

- Supplied Markdown copy has matching SHA-256:
  `4c492b4d94ea409dae668bf32fad03a0a8f76ceac6596abb5c772b48e01456bb`.
- New hand-authored Markdown passed whitespace checks; the copied source retains
  its original bytes and line endings.
- Checked timing/control/cohort arithmetic and cold/warm reporting requirements.
- Existing staged/unstaged changes were preserved; no staging or commit performed.
- Nineteen mathematical/gradient/optimizer/telemetry/report checks passed in the
  pinned Python3.10/PyTorch2.6.0 runtime. New FD-diffusion optimizer states and
  selected checkpoint losses match the historical implementation exactly in
  the controlled CPU test. Full-volume scientific/performance gates are pending.
- Tracked notebook-output checker passed; no source notebook outputs were added.

## 2026-11-10: implementation and runtime recovery

- Added isolated losses, continuous derivative operators, registration/evaluation
  workers, all-GPU controller and external CSV/JSON/Markdown/plot reporting.
- First preflight execution `oasis_reg6_20261110T0916Z_v1` retained all six failed
  startup attempts: the configured default container had lost GPU device access
  (NVML Unknown Error / no CUDA GPUs available). No optimizer update occurred.
- Preserved the old container and launched a separate experiment runtime from
  the identical pinned image with read-only repo/raw-data and writable artifact
  binds. Actual container/host paths remain in private configuration and external
  execution manifests. All four devices and a tiny CUDA operation passed.
- Corrected telemetry handling of exited processes without a VmRSS field and
  ensured active-slot status cleanup also runs after monitor errors.
- New execution `oasis_reg6_20261110T0917Z_v2` reruns all preflight gates under
  the corrected runtime and a fresh source freeze. Earlier artifacts are retained.
- Preflight accepted2026-11-10T09:17:23Z:19 tests and all six full-resolution
  smokes with separate same-GPU field/loss/metric replay. All attributed NVML
  and RSS traces passed. Maximum reserved training memory2.344GiB; measured
  artifact projection127.4GB includes30%margin, with ample external capacity.
- The durable all-phase controller advances through coarse pilots, separate
  evaluation waves, bounded extensions, frozen per-arm first10 grids and reports.
- Production controller PID/start identity and exact invocation are external in
  `controller.json` / `production_launch.json`; stdout is `controller_all.log`.
  Live `status.json` and review `progress.md` expose GPU assignments; detailed
  reports refresh after separate evaluation waves. No Git commit was performed.

## 2026-11-10: accepted completion and paired interpretation

- All116 registration/evaluation blocks and721 observations accepted, with zero
  v2 failure artifacts. Counts:6smokes+110coarse+605final;550 warm primary case
  rows and19250 primary label rows. No coarse range extensions were needed.
- Best tested mean-Dice weights are1 for both diffusion arms,0.3 for both BE,
  and0.03 for both KL arms. Analytic-KL0.03 is penultimate in its first10 grid;
  the rank0-selected window does not fully bracket the population optimum.
- Added a metadata-only paired analyzer outside the frozen numerical package.
  It verifies complete cohorts, original source hashes, all warm-case input
  receipts/backend flags and independently reproduced summary statistics.
  Case bootstrap uses20,000 paired resamples, seed20261110; exploratory,
  post-selection intervals are not corrected for multiple comparisons.
- The canonical label rule excludes a missing warped label34 in KL-FD0.03,
  rank1 (fixed22/moving6/warped0 voxels). Other selected nonzero arms retain
  all35 labels. Common-label sensitivity is reported separately.
- Detailed interpretation retains both Jacobian conventions and NDV: central
  folding substantially understates analytic sampled folding for FD-KL.
- Warm timing differences are decomposed into adaptive updates and loop cost.
  All55 final cold/warm rank0 pairs have identical Dice and update counts.
  Registration memory and separately sampled scoring memory remain separate.
- Canonical detailed posthoc review: external `paired_review_v3/` beneath the
  execution review root, with CSV/JSON, Markdown and four PNG/PDF figure sets.
  Earlier original reports and incomplete posthoc snapshots remain preserved.
