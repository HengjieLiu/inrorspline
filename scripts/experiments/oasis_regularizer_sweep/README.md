# OASIS six-arm regularizer experiment

The numerical protocol and resource definitions are frozen in
`experiments/2026-11-10_oasis_regularizer_fd_analytic_sweep/codex_execution_spec.md`.
The package does not change historical registration code.

Resolve runtime, OASIS and artifact mappings from ignored private configuration.
The host-only controller imports no Torch and performs no image processing;
registration, evaluation, tests and reports run in the specified Docker container.

```bash
python3 -B -m scripts.experiments.oasis_regularizer_sweep.controller \
  --container "$CONTAINER" \
  --artifact-host "$ARTIFACT_ROOT" \
  --artifact-container /database/inrorspline \
  --oasis-root "$OASIS_ROOT_CONTAINER" \
  --run-id "$RUN_ID" --phase preflight
```

Use the same command with `--phase all` to run/resume the frozen execution after
preflight. It runs coarse cold/warm pilots, separate metrics, bounded range
extensions, selection, then the frozen first10 blocks and final reports.
All visible GPUs are used with one child per GPU and same-GPU replay.
`--phase report` regenerates external reports only, without registration.

Source changes require a new run ID; never overwrite failed execution evidence.
Incomplete worker handshakes fail closed. An active-worker reconciliation marker
requires an identity audit before launching another child on its GPU.
Do not start two controllers for one run; its file lock prevents this.

Read `<artifact-root>/runs/<run-id>/status.json` and the external review
`progress.md` while running. `report.md` includes all available timing/memory
settings; CSV/JSON companions include complete case, stage and label records.
No Git staging, commits, paper changes, sync or unrelated process termination
is part of this workflow.
