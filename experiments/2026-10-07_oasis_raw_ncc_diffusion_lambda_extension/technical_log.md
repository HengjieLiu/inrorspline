# Technical log

## Known limitations

All first10 cases participate in this explicitly requested follow-up tuning.
Ranks1–9 are no longer held-out confirmation for selection among these lambdas.
Runtime is diagnostic under existing workloads; global bitwise CUDA determinism
is not guaranteed. P40's original outputs and search ceiling remain unchanged.

## Implementation

Reused the prior numerical objective, optimizer, metrics and four-GPU scheduler.
Added a separate bounded 30-fit launcher with read-only source integrity checks
and combined casewise/mean reports. Worker gains an optional experiment-variant
metadata field; default behavior and numerical implementation remain unchanged.

Preflight, actual launches and completion are recorded in external manifests,
status, launcher/worker logs and acceptance reports. No success is assumed here.

## CPU and source preflight

- Six new extension contracts and all 14 existing refinement contracts pass.
  The existing suite uses its direct function-test harness, not plain unittest
  discovery (which collects zero tests for that older function-style module).
- Numerical objective, common metrics and prefix code hashes match cached
  preparation. Scheduler hashes match all ten actual accepted lambda30 runs.
  The earliest preparation predates P40 scheduler/report fixes; scheduler
  identity is therefore checked against production controls, not preparation.
- Current GPU memory/utilization is recorded at launch rather than copied from
  P40. The same approved image/digest and UID:GID are recorded in each invocation.

## GPU preflight and launch

- Full-volume rank0/lambda1000 one-step smoke passed, including exact dense
  initialization, finite serialized DVF, canonical metrics and completion marker.
- Launched 30 production fits on 2026-10-07 PDT, one new worker/GPU0–3. Prior
  fields, caches and reports are read-only inputs. No unrelated jobs are altered.
- Launcher automatically validates every completed candidate and produces the
  combined CSV/JSON/Markdown report only after all 30 succeed.

## Completion and independent acceptance

- Completed 2026-10-07 19:49 PDT (2026-10-08 02:49 UTC). All 30 production
  candidates and the smoke completed on attempt0, without OOM or failed runs.
- Final validation passed case/direction/backend metadata, finite serialized
  fields, exact prefix hashes, full-volume raw-NCC/diffusion identity, completion
  markers and unchanged prior-artifact checksums. No sweep workers remain.
- Independently recomputed every exported cohort mean from the 70 casewise
  rows, verified all 14 means within1e-12, and matched both prior lambda30
  subset means within1e-12. Source-only notebook and whitespace audits passed.
- First10 mean Dice/HD95/ASD: lambda100 `0.826171/1.892240/0.370711`,
  lambda300 `0.800950/1.986010/0.436688`, lambda1000
  `0.748155/2.251212/0.595842`. Existing lambda30 remains selected by
  mean Dice then ASD/HD95; lambda100 has the best raw-NCC mean HD95.
- Mean foreground nonpositive-Jacobian percentage drops from2.925658%
  (lambda30) to1.304788% (100),0.260561% (300),0.023623% (1000).
  Stronger smoothing at300/1000 comes with substantially worse accuracy.
- Lambda100 increases simultaneous per-case VFA tolerance matches from4/10
  to6/10, but no raw-NCC cohort mean meets all three tolerances. This remains
  exploratory first10 tuning, not unbiased validation.
