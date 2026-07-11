---
doc_type: technical_log
status: complete
date_started: 2026-06-25
date_completed: 2026-06-25
---

# LungCT DirectCP 4-Stage Global NCC + Dense FD-BE All-Case Sweep Log

## Known Limitations

- This all-case sweep reuses completed case `01` and `08` rows from the
  selected-case FD-BE diagnostic.
- The objective is intentionally not pTVReg-matched: it uses full-crop global
  NCC and dense forward-difference BE, while pTVReg uses local CC and isoTV.
- Dense forward-difference BE is not identical to analytic cubic B-spline BE.
- Alpha values are absolute dense-BE weights and should not be compared directly
  with IDIR bending weights or pTV isoTV weights.

## Protocol

```text
method = direct_cp_32_16_8_4_global_ncc_fd_be
cases_reused = 01,08
cases_launched = 02,03,04,05,06,07,09,10
alphas = 0,10,100,1000,10000,100000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 0,1,2,3
```

## Verification Log

- Passed: compile/import check for launcher, aggregator, runner, and common
  helper modules inside `py310sitreg_v1_test1`.
- Passed: loader audit for cases `01-10`; images, landmarks, and fixed T00 lung
  masks are present on the cropped 1 mm grid.
- Passed: launcher dry run queues exactly `48` missing-case jobs and skips the
  reused case `01` and `08` rows.
- Passed: smoke job on missing case `02` at alpha `100000` with stage epochs
  `1,1,1,1`; output includes `summary.json`, dense DVFs,
  `per_landmark_tre.csv`, `stage_history.csv`, and `prolongation_metrics.csv`.
  The smoke recorded the intended 4-stage schedule, absolute alpha, and
  full-crop global NCC + dense forward-difference BE objective.
- Passed: full missing-case launch completed `48/48` jobs with return code 0.
- Passed: aggregation merged reused cases `01,08` with the new missing-case
  outputs and produced `60/60` FD-BE rows, `160` comparison rows, and four
  all-case ellipse-style comparison plots.

## Result Notes

- Best FD-BE mean point by original-snap TRE: alpha `10000`, original snap TRE
  `0.966 +/- 0.151 mm`, foreground stdLogJ `0.185452`, and foreground
  `J <= 0` `0.0037%`.
- Best pTV DirectCP comparison point: isoTV weight `0.11`, original snap TRE
  `0.922 +/- 0.152 mm`, foreground stdLogJ `0.179961`, and foreground
  `J <= 0` `0.0138%`.
- Best IDIR BE comparison point: alpha bending `10`, original snap TRE
  `1.017 +/- 0.176 mm`, foreground stdLogJ `0.142869`, and foreground
  `J <= 0` `0.0000%`.
