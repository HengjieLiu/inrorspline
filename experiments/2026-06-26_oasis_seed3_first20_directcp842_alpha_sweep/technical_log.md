# OASIS DirectCP 8-4-2 Remaining Alpha Sweep Technical Log

## Known Limitations

- This run completes the DirectCP `8,4,2` alpha curve only; it does not modify
  the existing five-sweep baselines or the previous saved-baseline figure
  package.
- Alpha `1000` is reused from the completed preliminary first20 root and is not
  relaunched in this step.
- Comparisons remain scoped to the OASIS seed-3 first10/first20 protocol.

## 2026-06-26 Setup

- Added remaining-alpha launcher for alphas `0,10,100,10000,100000`.
- Added full DirectCP `8,4,2` aggregator that combines the remaining-five-alpha
  root with the reused alpha `1000` root.
- Added a new plus-DirectCP842 figure script that extends the existing
  saved-baseline paper-style plotting package while writing to a new output
  directory.
- Registry entry P1i records the output roots, reused alpha handling, launch
  settings, and acceptance checks.

## Verification Notes

Completed checks:

- Docker compile passed for the remaining-alpha launcher, full-sweep aggregator,
  plus-DirectCP842 plot script, `run_one.py`, and `common.py`.
- Rank-0 alpha-0 smoke with stage epochs `1,1,1` completed and wrote stage
  history, stage summaries, prolongation diagnostics, dense field, and
  regularity metrics.
- Launcher dry-run queued exactly `100` jobs, ranks `0-19`, alphas
  `0,10,100,10000,100000`, GPUs `0,1,2,3`, no alpha `1000`, with balanced
  dry-run GPU assignment.
- Full launch completed `100/100` new remaining-alpha jobs with `0` skipped and
  `0` failures; launcher elapsed time was `6696.98` seconds.
- Aggregation combined the new remaining-five-alpha root with the reused
  alpha `1000` root and validated `100` remaining summaries, `20` reused
  alpha1000 summaries, `120` combined summaries, and `errors=[]`.
- Figure package generation passed with first10 sweep rows `360`, first20 sweep
  rows `720`, population rows `41` per cohort, validation errors `[]`, and
  `162` non-empty PDF/PNG files per cohort.

## Result Summary

- Reused alpha `1000` remained the best mean-Dice point for DirectCP `8,4,2` on
  both first10 and first20.
- Alpha `10000` had the best mean HD95 on both first10 and first20.
- First10 DirectCP `8,4,2` alpha `1000`: Dice `0.8054`, HD95 `2.2429`,
  symmetric ASD `0.4507`, foreground `jdet <= 0` `4.3015%`.
- First20 DirectCP `8,4,2` alpha `1000`: Dice `0.8077`, HD95 `2.2589`,
  symmetric ASD `0.4531`, foreground `jdet <= 0` `4.4071%`.
- New figure root:
  `experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/pareto_first10_first20_alpha5_saved_baselines_plus_directcp842_paper_v1/`.
