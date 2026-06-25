# OASIS Alpha Sweep Pre-Run Review

- Generated: `2026-06-25 11:26:30 +0000`
- Large output root: `/database/inrorspline/runs/oasis_seed3_case0_prerun_be_1000e_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_case0_prerun`
- Expected runs: `12`
- Completed summaries: `12`
- Case: OASIS seed-3 shuffled rank 0, original sorted index `158`, fixed/moving `(0333, 0115)`
- Foreground rule: fixed raw image `> 0`
- Native analytic regularity: IDIR uses coordinate-autograd BE/diffusion; spline methods use B-spline analytic BE/diffusion.
- Contact sheet plots native analytic BE/diffusion; spline-only analytic columns remain in CSV/JSON for audit.

## Acceptance

- PASS: all expected runs are present with the expected case IDs.

## Best Alpha By Method

- IDIR: best Dice alpha `1`, best HD95 alpha `1`, best symmetric ASD alpha `1`.
- SINR cps4: best Dice alpha `10000`, best HD95 alpha `10000`, best symmetric ASD alpha `10000`.
- SINR cps2: best Dice alpha `10000`, best HD95 alpha `10000`, best symmetric ASD alpha `10000`.
- DirectCP cps4 lr1e-3: best Dice alpha `10000`, best HD95 alpha `10000`, best symmetric ASD alpha `10000`.

## Files

- `summary_runs.csv`
- `summary_runs.json`
- `loss_history_all.csv`
- `pareto_contact_sheet.pdf`
- `loss_curves.pdf`

