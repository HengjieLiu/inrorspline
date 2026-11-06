# P81 multi-GPU scheduling continuation

Date: 2026-11-05, America/Los_Angeles. User-authorized runtime update to the
already-running numerical protocol v1.3. This addendum supersedes the earlier
GPU0-only / eight-worker / no-retry scheduling rules, not the scientific fit.

Activated2026-11-05 around10:26:55UTC (02:26:55PST). Handoff preserved93
completed jobs and adopted eight active workers without signalling any worker.
All94 CPU tests and serialized native-volume smokes on GPUs0/1/2/3 passed.
Initial new-controller snapshot had13 running workers:8 on GPU0,4 on GPU2,
1 on GPU3. GPU1 was waiting for external memory occupancy to clear. These
are time-stamped snapshots; the current counts are in `queue_state.json`.

## Scheduling contract

| Physical GPU | Maximum P81 workers |
| --- | ---: |
| 0 | 16 |
| 1 | 16 |
| 2 | 4 |
| 3 | 4 |

Use one shared pending queue, with no permanent GPU affinity. If a case does
not fit one GPU, try it on the others; smaller eligible cases may backfill.
Determine the largest feasible total-worker tier from16/8/4/2/1 separately
for each GPU, within that GPU's maximum. Account for all externally occupied
memory, the unallocated part of each active worker's profiled peak reservation,
and2GiB device headroom. Recheck free memory before each GPU batch and each
individual spawn. A memory drop may stop a batch early; never kill active
workers merely to reach a smaller tier.

Poll every60 seconds. A fully occupied GPU keeps waiting and does not prevent
admissions elsewhere. Pending jobs are not abandoned after a fixed occupancy
deadline. Host RAM/disk guards and the existing per-worker stall/wall-time
watchdogs remain active. Limits apply to P81 workers; other users' processes
are accounted for through free-memory telemetry and are never signalled.

## OOM handling

CUDA OOM attempts are recorded in the failure ledger, with their logs and
partial outputs preserved. Automatically return the case to the shared queue,
increase its reservation by at least2GiB or25%, and impose a60-second cooldown.
Repeated OOM cooldown doubles to a15-minute maximum; the case remains eligible
on every authorized GPU. Normal pending cases precede repeated OOM retries.
Reservations are capped below the largest device's capacity so an idle device
can eventually retry the case. Genuine capacity failures may remain retrying;
retry counts and waiting reasons stay visible. Other failures remain terminal
for that job while independent work continues.

Each retry gets a new attempt directory. Already-published tree-budget
checkpoints and their dependent jobs remain valid. A retry rebuilds the path
deterministically; at a published budget it verifies geometry and checksums,
restores the exact saved coefficients, and continues. Geometry mismatch fails
explicitly rather than mixing incompatible paths. Count earlier failed-attempt
costs separately when analyzing timing; published results are not overwritten.

## Implementation and handoff

New runtime code lives in `scripts/experiments/oasis_polyaffine_scheduler/`.
The entire original `oasis_polyaffine_fieldfit/` directory, scientific
configuration, original launch manifest and all accepted inputs remain
byte-identical. The GPU-selecting entry point calls the existing `Worker`
implementation with the same seeds, optimizer settings, masks and metrics.
Mask workers by the assigned physical GPU UUID, with logical CUDA device0.

Generated continuation records live under
`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v3/scheduler_revisions/multigpu_v1/`.
Keep CPU-test receipts, native-volume GPU smokes, source archive, the new
scheduler manifest, old-controller identity and exact pre-handoff queue snapshot
there. The current queue remains at the original `queue_state.json` path.
`ACTIVE_SCHEDULER.json` binds the active runtime revision.

Handoff stops only the old controller, takes the same controller lock and adopts
its still-running workers on GPU0. It never stops a worker or reruns a completed
job. New attempts record physical GPU, UUID, scheduler revision and OOM history.
Numerical results remain comparable under the same protocol; wall times are
from different concurrent GPU loads and are not exclusive-hardware benchmarks.

Before handoff: require unchanged scientific-source hashes, all scheduler CPU
tests, and a serialized native-volume SVF smoke on each GPU. Original numerical
and input-acceptance receipts remain bound to the unchanged scientific sources.

Resume command inside the configured container:

```bash
python -B /workspace/repo/scripts/experiments/oasis_polyaffine_scheduler/dispatcher.py \
  --root "${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v3" \
  --revision "${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/execution_20261105_v3/scheduler_revisions/multigpu_v1/manifest.json"
```

Use the same configured artifact-writing container identity. The shared lock
prevents simultaneous controllers. Do not resume the historical GPU0-only
controller after this continuation is active.
