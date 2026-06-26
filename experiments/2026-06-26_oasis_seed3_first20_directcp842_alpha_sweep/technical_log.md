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

Pending checks:

- Docker compile of changed scripts.
- Rank-0 alpha-0 smoke with stage epochs `1,1,1`.
- Launcher dry-run: exactly `100` jobs, ranks `0-19`, alphas
  `0,10,100,10000,100000`, GPUs `0,1,2,3`, no alpha `1000`.
- Full launch completion: `100/100` new remaining-alpha `DONE` jobs.
- Aggregation: `100` remaining summaries, `20` reused alpha1000 summaries,
  `120` combined summaries, and no validation errors.
- Figure package: first10 sweep rows `360`, first20 sweep rows `720`,
  population rows `41` per cohort, and non-empty PDF/PNG outputs.
