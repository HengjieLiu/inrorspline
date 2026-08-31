# Execution specification: dedicated four-GPU all-case P2e two-window audit

## Implemented entry point

`scripts/experiments/learn2breath_mrdbscp_p2e/run_pairlocal.py`

The runner is intentionally separate from the historical pTV crop loader. It
reuses only its low-level cubic B-spline, pTV local-CC, isoTV, and exact
prolongation routines.

## Docker invocation contract

Run in the Learn2Breath development container with externally provided roots
for COPD pair-local CT, Learn2Breath validation CT/lobes, R231 masks, runtime
artifacts, and the Learn2Breath evaluator repository.

```text
PAIRLOCAL_ROOT : challenge-like COPD pair-local data root
R231_ROOT      : prior R231 audit mask directory
RUNTIME_ROOT   : external experiment-artifact root
```

The all-case controller creates an immutable manifest of four persistent
workers: COPD/P2e-equivalent, COPD/Learn2Breath-lung, validation/P2e-
equivalent, and validation/Learn2Breath-lung. Each worker owns ten sequential
cases on one GPU, records one cold and nine warm cases, and waits after its
cold case until all four cold gates pass.

Preflight precedes the full registration and validates all forty rows. The
controller records optimizer-only, full-case-with-evaluation, worker-case,
worker-total, and overall-makespan timing. It does not submit a challenge ZIP
or select a winning window.

```bash
python /homebase/code_sync/inrorspline/scripts/experiments/learn2breath_mrdbscp_p2e/run_all_two_window_audit.py make-manifest \
  --out-root "$RUNTIME_ROOT" --copd-data-root "$PAIRLOCAL_ROOT" \
  --validation-data-root "$L2B_DATA_ROOT" --r231-root "$R231_ROOT" \
  --learn2breath-repo "$L2B_REPO"
```

The subsequent `preflight-all`, `launch`, and `aggregate` subcommands consume
the fixed manifest. Outputs are outside Git. Do not pass any landmark, R231,
or lobe-label asset to the loss.

## Acceptance checks

1. All forty preflights report the shared grid; COPD reports 300 in-bounds IDs
   and R231 agreement, while validation reports five aligned nonempty lobes.
2. COPD TRE uses physical world coordinates and trilinear fixed-grid samples.
3. All fields reload as `float32 [3,X,Y,Z]` with fixed CT affine; headline
   folding uses the unchanged Learn2Breath evaluator for both cohorts.
4. The two all-case notebooks execute in Docker and display every 2x9 panel
   inline with the accepted coronal orientation.
5. Aggregation refuses missing/nonfinite rows and preserves each lobe Dice.
