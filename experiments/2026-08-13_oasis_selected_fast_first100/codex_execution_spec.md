# OASIS100 Selected Fast-Schedule Evaluation Execution Spec

## Summary

Evaluate the five selected accelerated OASIS schedules on ranks `10-99` and reuse the already evaluated ranks `0-9` from the unified first10 sweep.

Selected schedules:

- `MR_A_fast_decay`
- `D2_A_bal_lr10x`
- `D4_A_bal_lr10x`
- `S2_A_bal_lr10x`
- `S4_A_bal_lr10x`

The resulting review report combines `5 x 10` reused first10 rows and `5 x 90` new rows for `500` total selected OASIS100 rows.

## Execution

- Review root: `experiments/2026-08-13_oasis_selected_fast_first100/`
- New output root: `${DATA_ROOT}/runs/oasis_selected_fast_first100_alpha1000_v1`
- First10 reuse root: `${DATA_ROOT}/runs/oasis_unified_acceleration_sweep_alpha1000_v1`
- Launcher: `scripts/experiments/oasis_alpha_sweep/launch_oasis_selected_fast_first100.py`
- Aggregator: `scripts/experiments/oasis_alpha_sweep/aggregate_oasis_selected_fast_first100.py`

Run order:

1. Syntax check touched scripts.
2. Dry-run full manifest.
3. Smoke rank `10` for all five selected schedules with one epoch.
4. Run `MR_A_fast_decay` data-parallel across four rank chunks:
   - GPU `0`: ranks `10-32`
   - GPU `1`: ranks `33-55`
   - GPU `2`: ranks `56-77`
   - GPU `3`: ranks `78-99`
5. After MR completes, run the remaining four methods model-parallel:
   - GPU `0`: `D2_A_bal_lr10x`, ranks `10-99`
   - GPU `1`: `D4_A_bal_lr10x`, ranks `10-99`
   - GPU `2`: `S2_A_bal_lr10x`, ranks `10-99`
   - GPU `3`: `S4_A_bal_lr10x`, ranks `10-99`
6. Aggregate first10 reused rows plus new ranks `10-99`.

## Outputs

- `selection_summary.md`
- `selected_fast_oasis100_case_results.csv/json`
- `selected_fast_oasis100_summary.csv/json`
- `selected_fast_oasis100_reference_runtime.csv`
- `selected_fast_oasis100_report.md`
- `missing_or_failed_jobs.json`

Runtime is recorded as reference-only because GPUs were shared/occupied. Final paper benchmark timing should be rerun later under isolated persistent-process conditions.
