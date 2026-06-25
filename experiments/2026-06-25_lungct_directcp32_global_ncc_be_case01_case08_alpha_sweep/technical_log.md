---
doc_type: technical_log
status: active
date_started: 2026-06-25
---

# LungCT DirectCP32 Global NCC + Analytic BE Sweep Log

## Known Limitations

- This is a selected-case diagnostic on cases `01` and `08`, not an all-case
  claim.
- The objective is intentionally not pTVReg-matched: it uses full-crop global
  NCC and analytic cubic B-spline BE, while pTVReg uses local CC and isoTV.
- Alpha values are absolute analytic-BE weights and should not be compared
  directly with IDIR bending weights or pTV isoTV weights.
- Training is in the cropped 1mm pTV-prep grid, while the headline metric is
  `original_snap` TRE after mapping the dense field back to the native crop.

## Protocol

```text
method = direct_cp_32_16_8_4_global_ncc_bspline_be
cases = 01,08
alphas = 0,0.1,1,10,100,1000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 2,3
```

Training loss:

```text
loss = full_crop_global_NCC + alpha * analytic_cubic_bspline_BE
```

Fixed T00 lung mask is used only for foreground regularity metrics.

## Verification Log

- Passed: compile/import check for `common.py`, `run_one.py`, launcher, and
  aggregator inside `py310sitreg_v1_test1`.
- Passed: loader audit for cases `01,08`; both cases loaded and preprocessing
  verification passed.
- Passed: launcher dry run queued exactly `12` jobs over GPUs `2,3`.
- Passed: smoke jobs for case `01`, alphas `0` and `1000`, stage epochs
  `1,1,1,1`; outputs included `summary.json`, `stage_history.csv`,
  `prolongation_metrics.csv`, `per_landmark_tre.csv`, and dense DVF arrays.
- Smoke exact-knot preservation error was near numerical precision:
  stage `32->16`, `16->8`, and `8->4` exact-knot mean absolute preservation
  errors were around `4.5e-11`, `7.2e-11`, and `1.0e-10`.
- Runtime blocker: smoke timing was too high for the requested full schedule.
  Case `01`, alpha `0`, stage epochs `1,1,1,1` took `149.5 sec`; alpha
  `1000` took `186.4 sec`. Stage timings for alpha `1000` were approximately
  `43.8`, `7.5`, `19.8`, and `21.6 sec` for stages `32`, `16`, `8`, and `4`.
  Launching the full `200,400,800,1000` schedule as-is would likely require
  many hours per job and multiple days for all `12` jobs on two GPUs.
- Full launch status: not launched pending an implementation/runtime decision.

## Forward-Difference BE Speed Check

After the analytic B-spline BE smoke proved too slow, a non-mutating benchmark
tested a dense forward-difference BE training objective on GPU `2`. The timed
step included control-grid expansion, moving-image warp, full-crop global NCC,
dense forward-difference BE on the dense displacement field, backward, and Adam
step.

Case `01` timings:

| Stage | Downsample | Eval shape | NCC only sec/iter | NCC + fd-BE sec/iter | Peak GB |
|---:|---:|---|---:|---:|---:|
| 32 | 8 | 27x19x29 | 0.0042 | 0.0102 | 0.14 |
| 16 | 4 | 53x38x58 | 0.0045 | 0.0100 | 0.14 |
| 8 | 2 | 105x75x116 | 0.0043 | 0.008-0.009 | 0.22 |
| 4 | 1 | 210x150x232 | 0.0062 | 0.0480 | 1.30 |

Case `08` timings:

| Stage | Downsample | Eval shape | NCC only sec/iter | NCC + fd-BE sec/iter | Peak GB |
|---:|---:|---|---:|---:|---:|
| 32 | 8 | 29x26x32 | 0.0043 | 0.0051 | 0.23 |
| 16 | 4 | 57x52x64 | 0.0040 | 0.0090 | 0.23 |
| 8 | 2 | 114x104x127 | 0.0041 | 0.0119 | 0.36 |
| 4 | 1 | 228x207x254 | 0.0087 | 0.0782 | 2.12 |

Estimated pure training-step time for the requested `200,400,800,1000`
schedule is roughly `1-1.5 min/case` before run setup, prolongation diagnostics,
final dense-field saving, TRE, and regularity evaluation. This is feasible and
is orders of magnitude faster than exact analytic B-spline BE training. It is,
however, a different training regularizer and should be labeled as dense
forward-difference BE, with analytic B-spline BE retained only as a posthoc
metric if needed.
