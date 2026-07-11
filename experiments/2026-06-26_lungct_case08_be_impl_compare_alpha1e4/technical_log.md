---
doc_type: technical_log
status: complete
date_started: 2026-06-26
date_completed: 2026-06-26
---

# LungCT Case08 DirectCP 4-Stage BE Implementation Comparison Log

## Known Limitations

- This is a selected-case diagnostic on case `08`, not an all-case conclusion.
- Both methods use the same numeric alpha `10000`, but dense forward-difference
  BE and analytic cubic B-spline BE can have different scales.
- The objective is full-crop global NCC and is not pTVReg-matched local CC +
  isoTV.

## Protocol

```text
case = 08
methods = direct_cp_32_16_8_4_global_ncc_fd_be,
          direct_cp_32_16_8_4_global_ncc_bspline_be
alpha = 10000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 2,3
```

## Verification Log

- Passed: compile/import check for runner and common helpers inside
  `py310sitreg_v1_test1`.
- Passed: loader audit for case `08`; shape `228,207,254`, nonzero fixed T00
  lung mask, 300 landmarks, preprocessing verification passed.
- Passed: minimal smoke for both BE variants with stage epochs `1,1,1,1`.
  Both wrote `summary.json`, dense DVFs, `per_landmark_tre.csv`,
  `stage_history.csv`, and `prolongation_metrics.csv`.
- Passed: full concurrent launch on GPUs `2,3`; both jobs completed with
  status `completed`.
- Passed: result comparison and report.

## Result Notes

- Dense FD-BE: original-snap TRE `1.087383` mm, original-subvoxel TRE
  `1.146554` mm, foreground stdLogJ `0.192354`, elapsed `351.897` seconds,
  peak memory `6.492` GB.
- Analytic B-spline BE: original-snap TRE `1.087383` mm, original-subvoxel TRE
  `1.145162` mm, foreground stdLogJ `0.191389`, elapsed `367.985` seconds,
  peak memory `6.492` GB.
- Original-snap TRE is tied after snap-to-voxel rounding; analytic B-spline BE
  is slightly smoother and slightly better on subvoxel diagnostics, with about
  `16.1` seconds extra elapsed time.
