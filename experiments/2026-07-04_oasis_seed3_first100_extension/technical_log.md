# Technical Log

## Known Limitations

- The final publication figure generator is strict: a method/alpha/reference
  row is included only after all `100` cases are present and validated.
- The tracker is intentionally partial and may show first20-only curves before
  rank `20-99` jobs complete.
- Saved DL and Greedy displacement preflight stops if a rank `20-99` source
  field is missing.
- Missing fixed, moving, or warped label voxels are treated as `NaN` for Dice,
  HD95, and ASD; population means use finite-only `nanmean`.

## 2026-07-04 Implementation Notes

- Added generalized first100 constants and launch wrappers with skip-existing
  support for ranks `0-19`.
- Added multi-GPU launchers for saved baselines and single-stage sweeps.
- Reused the existing DirectCP 8-4-2 lr `1e-3` launcher for the MR-D-BSCP
  extension via the top-level orchestrator.
- Added a monitor that writes `progress_log.md`, appends tracker snapshots, and
  refreshes `tracker/oasis_first100_progress_tracker.png`.
- Added first100 aggregation wrapper for strict final paper-style plots under
  `first100_artifacts/oasis100/`.

## 2026-07-04 Verification

- Docker `py_compile` passed for changed OASIS sweep scripts and the metric
  regression test.
- Synthetic metric regression passed: absent-label Dice/HD95/ASD are `NaN`,
  and `safe_nanmean` uses finite values only.
- Saved-baseline dry-run passed preflight and produced `400` rank20-99 launch
  commands.
- MR-D-BSCP 8-4-2 dry-run produced `480` rank20-99 launch commands.
- Single-stage dry-run produced `2400` rank20-99 launch commands.
- Top-level dry-run produced `3280` launch commands and kept aggregation as
  dry-run-only.
- Smoke roots:
  - `/database/inrorspline/runs/oasis_seed3_first100_extension_saved_baseline_smoke_v1`
  - `/database/inrorspline/runs/oasis_seed3_first100_extension_smoke_v1`
- Smoke result: `7/7` smoke summaries have `DONE` markers and finite Dice,
  HD95, and foreground `std(log J)` core metrics.

## 2026-07-04 Production Launch

- Production top-level launch started at `2026-07-04 04:53:36 +0000`.
- Launch command:
  `python scripts/experiments/oasis_alpha_sweep/launch_oasis_first100_extension.py --stage all --gpus auto --max-concurrent 0`.
- Top-level launcher log:
  `experiments/2026-07-04_oasis_seed3_first100_extension/launch_all_20260703_215334.log`.
- Latest process snapshot:
  `experiments/2026-07-04_oasis_seed3_first100_extension/production_processes_latest.txt`.
- The first active stage is saved-baseline/reference evaluation on all four
  visible GPUs. The live monitor is active and refreshes `progress_log.md` and
  `tracker/oasis_first100_progress_tracker.png` every 5 minutes.

## 2026-07-04 Hourly Watchdog

- Started an hourly watchdog inside `py310sitreg_v1_test1` with interval
  `3600` seconds.
- Watchdog script:
  `scripts/experiments/oasis_alpha_sweep/watch_oasis_first100_hourly.py`.
- Watchdog log:
  `experiments/2026-07-04_oasis_seed3_first100_extension/hourly_watchdog.md`.
- Watchdog snapshots:
  `experiments/2026-07-04_oasis_seed3_first100_extension/hourly_watchdog_snapshots.jsonl`.
- Watchdog stdout:
  `experiments/2026-07-04_oasis_seed3_first100_extension/hourly_watchdog.stdout.log`.
- Behavior: refresh the standard monitor/tracker, write process snapshots, check
  launcher-state failures and running status, and restart the top-level
  first100 launcher with skip-existing behavior if the launcher disappears
  before all `3280` new jobs complete.
- Initial watchdog check at `2026-07-04 05:04:35 +0000`: top-level launcher
  running, `47/3280` new jobs done, `4` running, `0` failed.
