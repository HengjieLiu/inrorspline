# Technical Log: LungCT FD-BE cps4 Alpha Landscape

## Known Limitations

- This is not a masked-loss experiment. The fixed T00 lung mask is used only for
  diagnostics/evaluation.
- Alpha values are FD-BE-family operating points and are not comparable as raw
  weights to IDIR bending energy, pTVReg isoTV, or Dual-INR loss weights.
- The `1e5` and `1e6` rows are reused from the accepted prior extension run;
  only `1e4`, `2e4`, `5e4`, `2e5`, and `5e5` are newly launched here.
- The rerun `1e4` rows supersede the old v011 `1e4` rows in v012 only if they
  pass validation. The old rows remain audit context.

## Execution Notes

- Planned 2026-08-14 by Codex.
- Source commit at planning: `68d2c4cc02ae465f2a9b23dfb965ee46ca864f41`
  on branch `gpu8`, with a dirty worktree.
- Docker environment: `hengjieliu96/dl_dir:py310sitreg_v1` via
  `py310sitreg_v1_test1`.
- Review directory:
  `experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/`.

## Preflight Status

- Docker `py_compile` passed for the landscape launcher, aggregator, casewise
  report builder, and v012 figure builder.
- Dry-run Wave 1 planned `80` jobs.
- Dry-run Wave 2 planned `20` jobs.
- Smoke Wave 1 completed `8/8` one-epoch jobs.
- Smoke Wave 2 completed `2/2` one-epoch jobs.
- After smoke completion, `nvidia-smi` temporarily reported stale `[Not Found]`
  compute entries holding roughly `40 GB` on each assigned GPU. GPU reset was
  not performed. The allocations cleared before the full launch, and a fresh
  `nvidia-smi` check showed no listed compute processes.

## Execution Outcome

- Wave 1 completed `80/80` jobs with no failures.
- Wave 2 completed `20/20` jobs with no failures.
- Aggregation accepted `100/100` current-source rows from the new run root.
- v012 combined FD-BE cps4 case table contains `140` rows:
  `7` alphas x `2` methods x `10` cases.
- The final FD-BE cps4 alpha values are `10000`, `20000`, `50000`, `100000`,
  `200000`, `500000`, and `1000000` for both INR-BSCP and D-BSCP.
- No masked-loss method IDs or masked-loss source paths are present in v012
  source rows.
- Alpha `1e4` rerun-vs-v011 audit passed with `0.0 mm` mean TRE delta and
  `0.0 mm` max case TRE delta for both FD-BE methods, so v012 is not
  review-only.
