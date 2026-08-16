# Technical Log

## Known limitations

- This refresh changes the plotted OASIS Panel A diagnostic domain only.
  OASIS training objectives remain unchanged.
- OASIS and DIR-LAB still use different anatomical masks because the datasets
  have different foreground definitions: fixed raw `image > 0` for OASIS and
  fixed T00 lung mask for DIR-LAB.
- v003 does not expand OASIS beyond the seed-3 shuffled first10 cohort.

## 2026-08-14

- Added paired OASIS Panel A NCC diagnostics to the OASIS runner.
- Added focused launcher:
  `scripts/experiments/fit_modes_population/launch_oasis_fixedmask_panelA_curves.py`.
- Added v003 renderer:
  `scripts/experiments/fit_modes_population/render_masked_panelA_both_datasets_diagnostics.py`.
- Dry-run resolved exactly 40 full-mode jobs.
- Smoke rank 0 passed across all four methods after fixing the single-stage
  spline diagnostic mask binding.
- Full run completed `40/40` records in the paired GPU schedule.
- v003 rendering passed with 160 source-index rows, exactly 40 OASIS Panel A
  replacements, and 26 generated images.
