# P80 technical log

## Known limitations and audited traps

- CUDA grid_sample backward may remain nondeterministic despite cuDNN flags.
- SameGPU replay required; shared workloads make timing diagnostic.
- Legacy bounded runners hardcode sixmaps / threelevels; use a new extension,
  explicit global lambda, levelindex3 for both new maps (not blockindex6/7).
- Legacy SVF settings must contain four paths only; controller flags never enter
  the path dictionary. New worker explicitly copies the four keys.
- Earlier roundoff recovery was scoped to ranks29/44 and cps8->4/cps4->2.
  Preserve those records; newv4 explicitly applies only to newcps2->1 transitions.
- Match sourceprefix separately from new targetmetadata; no old842 identity
  guard should reject a correctly extended8421 result.
- Earlier queue ETA counted failures as processed throughput. New ETA uses
  successful family-specific worker/audit times, with deferred repairs excluded.
- Preserve independent audit attempts in separate directories; worker DONE is
  not scientific acceptance. A report failure cannot invalidate completed fits.
- Fixed-positive foreground and absent-label canonical metrics unchanged.
- Artifact budget and native peak memory are measured from smokes before launch.

Implementation started after explicit approval; no historical sources or output
bytes changed. Production remains gated on tests, native smokes and replay.

## Preflight diagnostic recovery

The first fullsize smokes exposed an off-grid diagnostic-only autograd-context
error: the generic diagnostics wrapper had no_grad, while the retained-map
Jacobian needs gradients with respect to sampled coordinates. Optimization
completed; bounded diagnostics then failed. The queue deferred failures and
continued the other smokes, as designed. No production was launched.

Preserve execution_v1 and all source bytes. execution_v2 uses an isolated adapter
enabling gradients only inside the existing off-grid function, which detaches
all map fields. Losses, optimizers, stopping, checkpoints and refinement remain
unchanged. A real CPU off-grid regression reproduces the failure under no_grad
and verifies the fix; all nine native smokes and independent audits repeat.

## Verified pre-production gates

-53 tests passed (17 new contracts/numerics/failure tests,16 prior first100,
  11 prior transition-v3,6 ARC,3 diagnostic-context regression tests).
-Both historical rank29/44 witnesses reproduced with zero Adam updates.
-execution_v1 drained6 accepted/3 diagnostic failures; all artifacts retained.
-execution_v2 passed9/9 fullsize smokes and independent audits, zero failures.
-All2100 prefix selected checkpoints/fields verified by SHA256 and finite-shape
  scan (594.97s). Reuse of this integrity receipt in execution_v2 requires exact
  equality of prefix entries/cases/settings/protocol/bounds and numerical-source
  hashes; every production worker still rehashes its own source artifacts.
-Maximum reserved smoke memory: displacement2.277GiB, bounded2 4.627GiB,
  SVF3.922GiB. All three admission thresholds remain8GiB.
-Projected new artifact storage including30% margin:2.192e12 bytes.
-Renderer exercised with explicitly synthetic temporary data,12 exports and
  identical cross-depth axes; historical8421 appendix loader returns150 rows.

Active launch entry: `oasis_bspline8421_diagnostic_recovery.py --stop-after full`.
Production remains2100 new outputs; diagnostic smoke repeats are not candidates.

Production launched2026-10-17 05:35:31UTC (Oct16 22:35PDT), eight live workers
on slots0/0/1/1/2/2/3/3. Controller PID255150. Reports are automatic after queue
drainage; complete acceptance still requires all2100 independent replays.
