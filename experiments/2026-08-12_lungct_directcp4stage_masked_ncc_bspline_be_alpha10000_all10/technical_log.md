---
doc_type: technical_log
status: active
date_started: 2026-08-12
---

# LungCT Corrected MR-D-BSCP IDIR-Loss Test Log

## Known Limitations

- This is a single operating-point check at the current best FD-BE alpha
  `10000`, not a new regularization sweep.
- The input image protocol remains pTVReg-style 1mm landmark crop. It is not
  the native full-volume original IDIR protocol.
- The new regularizer is sampled analytic cubic B-spline BE at the same sampled
  fixed-mask coordinates used for the data term. This is intended to avoid the
  dense analytic-BE runtime bottleneck while preserving analytic spline
  derivative behavior.
- The loss uses sampled fixed T00 lung-mask global NCC, so stochastic variation
  is possible even with fixed seed.

## Protocol

```text
method = direct_cp_32_16_8_4_masked_ncc_bspline_be
cases = 01-10
alpha = 10000
lr = 1e-3
seed = 1
batch_size = 10000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
```

## Verification Log

- Passed: compile/import check for `common.py`, `run_one.py`, launcher, and
  aggregator inside `py310sitreg_v1_test1`.
- Passed: sampled B-spline derivative equivalence check against dense
  expansion on a toy grid; maximum absolute error was `1.788e-7`.
- Passed: case01 smoke with `stage_epochs=1,1,1,1`; summary and dense metrics
  were written successfully.
- Passed: launcher dry-run queued exactly `10` all-case jobs over GPUs
  `0,1,2,3`.
- Passed: full all-10 launch completed `10/10` jobs with return code `0`.
- Passed: aggregation generated `case_rows.*`, `mean_row.*`,
  `comparison_rows.*`, `manifest.json`, and `review_report.md` with
  completion errors `[]`.

## Result

Comparison at alpha `10000`:

| Row | TRE mean +/- SD | FG stdLogJ mean +/- SD | FG J<=0 % | FG BE | FG diffusion |
| --- | --- | --- | --- | --- | --- |
| Current full-crop NCC + dense FD-BE | 0.966 +/- 0.151 | 0.185452 +/- 0.060879 | 0.0037 | 0.001971 | 0.010033 |
| Corrected fixed-mask NCC + analytic B-spline BE | 1.037 +/- 0.153 | 0.265395 +/- 0.063547 | 0.0349 | 0.006341 | 0.016738 |

The corrected fixed-mask analytic-BE point is worse than the current FD-BE best
at the same alpha on both headline TRE and foreground stdLogJ. It should not
replace the current plotted `MR-D-BSCP (IDIR loss)` point without an additional
alpha sweep or a different sampling/regularization-scale decision.
