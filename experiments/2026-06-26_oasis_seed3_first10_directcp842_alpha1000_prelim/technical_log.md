# OASIS DirectCP 8-4-2 Alpha1000 Preliminary Technical Log

## Status

- Completed on 2026-06-26.
- User requested GPUs `2,3` only with max concurrency `2`.
- User noted GPU `2` may already have another job running but remains eligible.
- Mid-run extension requested after GPUs `0,1` became free: launch ranks
  `10-19` on GPUs `0,1` with max concurrency `2`, then report both first10
  and first20 comparison tables.

## Known Limitations Before Launch

- This is a single-alpha preliminary check at alpha `1000`, not the planned
  future six-alpha sweep.
- The original launch covered seed-3 first10; the extension launch adds ranks
  `10-19` under the same alpha/model setting.
- The final stage has cps2 spacing, but the nested control-grid margin follows
  the multistage knot-insertion implementation rather than the single-stage
  DirectCP cps2 control-grid shape exactly.
- The loss matches the existing OASIS spline objective, not the LungCT pTV LCC
  objective: full-image global NCC plus analytic cubic B-spline BE.

## Planned Checks

- Compile changed OASIS scripts inside Docker.
- Run rank-0 smoke with stage epochs `1,1,1`.
- Dry-run first10 launcher and verify exactly `10` jobs assigned only to GPUs
  `2,3`.
- Dry-run extension launcher and verify exactly `10` jobs assigned only to GPUs
  `0,1`.
- Run first20 total and aggregate separate first10 and first20 comprehensive
  comparison tables.

## Completion Notes

- Compile check passed inside Docker.
- Rank-0 smoke with stage epochs `1,1,1` passed.
- First10 launcher completed `10/10` ranks on GPUs `2,3` and first10
  aggregation passed with no errors.
- Extension launcher completed `10/10` ranks `10-19` on GPUs `0,1`.
- Final first20 aggregation passed with `20` new summaries and no errors.
- First10 new DirectCP `8-4-2` alpha `1000`: Dice `0.8054`, HD95 `2.2429`,
  symmetric ASD `0.4507`, foreground native BE `0.1137`, fold% `4.3015`.
- First20 new DirectCP `8-4-2` alpha `1000`: Dice `0.8077`, HD95 `2.2589`,
  symmetric ASD `0.4531`, foreground native BE `0.1150`, fold% `4.4071`.
