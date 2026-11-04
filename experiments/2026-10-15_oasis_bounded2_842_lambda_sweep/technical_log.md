# P76 technical log

## Limitations

Exploratory first-ten tuning; bounded maps do not guarantee zero folding for
the exported/interpolated field. Bounds/capacity stay fixed while diffusion
lambda varies. Two maps have twice the allowed optimizer budget per level.
CUDA backward nondeterminism prevents demanding bitwise cross-run identity.
Timing is diagnostic under concurrent workloads; allocated memory excludes
CUDA context/other processes, so reserved peaks and live GPU state also matter.

## Preflight audit

- User approved70 fresh fits and up to two workers on each GPU0–3.
- P70 hard-codeslambda1 in worker, validator and independent auditor. Its old
  queue aborts admissions after any failure; new isolated code must remove
  both restrictions without changing numerical definitions.
- Historical depth3 bounded-two cohort has49s mean optimizer time,74s mean
  worker time and2.50GiB peak allocated CUDA memory. These are not a promised
  ETA and exclude independent replay.
- P70 cross-depth prefix equality failed historically at cps2 due independent
  optimization variability. Reproduction differences will be diagnostic;
  reconstruction of each saved checkpoint remains a strict acceptance gate.

## Implementation and launch

- Added isolated explicit-lambda worker/auditor using the unchanged P70
  bounded objective and optimizer; new durable queue has eight slots,
  per-worker memory reservations, individual failure deferral, one deferred
  OOM retry, exclusive lock and live-child adoption on controller restart.
-24 worker/core regression tests passed. Five additional report/figure tests
  passed, including complete-matrix rejection, sample-SD arithmetic, correct
  C/F sweep selection, G's historical four-level point and row-specific axes.
- Three full-size rank0 smokes atlambda0.01/1/100 each completed six Adam
  updates, canonical serialization/metrics, objective value/gradient parity,
  contraction and100k off-grid probes, then independent replay on their GPUs.
  Maximum allocated/reserved smoke peak3.371GiB; admission remains8192MiB.
- Frozen source/input/reference hashes and canonical rank order verified.
  Production started2026-10-16 03:34UTC (local2026-10-15), two workers per
  GPU0–3 as1009:1020. GPU3 retains its unrelated active workload; no existing
  jobs were changed. Runtime reflects contention.
- A native sandboxed nvidia-smi invocation could not access the driver; live
  admission telemetry inside the approved container remained available.
  This was not a GPU worker/driver failure.

## Completion and acceptance

- All 70 production registrations and 70 independent replays completed
  successfully; no failures or OOM retries. Queue wall time 1344.36 s.
  Report accepted 2026-10-16 03:58:29 UTC (local October 15).
- Acceptance verified 420 map records, 2450 per-label records, 101920 history
  rows, 100 independently recomputed summary statistics, source integrity,
  finite fields, contraction/off-grid diagnostics and checkpoint consistency.
  Full histories checked contiguous updates, fixed LR, loss decomposition and
  lowest-loss checkpoint selection. Historical lambda-1 differences are saved
  as diagnostics, not silently replaced or required to be bitwise identical.
- Final regression coverage: 24 worker/core tests, 24 ARC tests (including
  historical renderer regressions), three report tests: 51 total passing.
- Lambda 1 maximizes mean Dice and minimizes mean HD95 and ASD within the
  sweep: 0.819183 / 1.764939 / 0.364375. FG SD(log J) 0.603642; folding
  0.000037%. Lambda 3/10/100 yield zero measured canonical foreground folding
  on these cases, but lower accuracy. No unconditional topology claim.
- Production peak allocated/reserved memory approximately 2.49/3.10 GiB.
  Lambda-1 mean optimizer/worker/replay times 67.66/96.41/23.50 s; shared GPU
  load remained active, so these are diagnostic timings rather than controlled
  speed measurements. Stage and per-case time/SD/range values saved externally.
- Updated C/F and retained four-level G rendered in a distinct arc_panels_v1
  release. Figure acceptance at 2026-10-16 04:02:08 UTC verified 55 settings,
  550 displayed rows (430 unique source rows), 550 mean/SD values and 18 axes.
  All source metrics and hashes match; PNG/PDF/SVG and embedded fonts checked.
  All three figures visually reviewed: readable labels, no clipped points or
  ellipses. Existing ranges needed no expansion. F/G retain tighter top rows.
- New seven-page A–G PDF assembled with original A/B/D/E pages unchanged;
  extracted page text matches originals. Historical reports, source numerics
  and plots preserved. All generated files stay in the external artifact store.
