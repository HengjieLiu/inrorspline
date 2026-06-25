---
doc_type: technical_log
status: active
date_started: 2026-06-25
---

# LungCT DirectCP32 Global NCC + Dense FD-BE Sweep Log

## Known Limitations

- This is a selected-case diagnostic on cases `01` and `08`, not an all-case
  claim.
- The objective is intentionally not pTVReg-matched: it uses full-crop global
  NCC and dense forward-difference BE, while pTVReg uses local CC and isoTV.
- Dense forward-difference BE is not identical to analytic cubic B-spline BE.
  Alpha values are absolute dense-BE weights and should not be compared directly
  with IDIR bending weights, pTV isoTV weights, or analytic B-spline BE weights.
- Training is in the cropped 1mm pTV-prep grid, while the headline metric is
  `original_snap` TRE after mapping the dense field back to the native crop.

## Protocol

```text
method = direct_cp_32_16_8_4_global_ncc_fd_be
cases = 01,08
alphas = 0,0.1,1,10,100,1000,10000,100000,1000000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 2,3
```

Training loss:

```text
loss = full_crop_global_NCC + alpha * dense_forward_difference_BE
```

Fixed T00 lung mask is used only for foreground regularity metrics.

## Verification Log

- Passed: compile/import check for `common.py`, `run_one.py`, launcher, and
  aggregator inside `py310sitreg_v1_test1`.
- Passed: loader audit for cases `01,08`; both cases loaded and preprocessing
  verification passed. Case `01` shape is `210x150x232`, and case `08` shape
  is `228x207x254`; both have 300 landmarks and nonzero fixed lung masks.
- Passed: launcher dry run queued exactly `12` jobs over GPUs `2,3`.
- Initial smoke bug/fix: alpha `1000` smoke exposed that forward-difference
  second-derivative tensors have different cropped shapes, so they cannot be
  accumulated into one dense per-voxel map. The training helper now computes the
  dense FD-BE objective as a scalar sum of per-term means.
- Passed: smoke jobs for case `01`, alphas `0` and `1000`, stage epochs
  `1,1,1,1`; outputs included `summary.json`, `stage_history.csv`,
  `prolongation_metrics.csv`, `per_landmark_tre.csv`,
  `disp_norm_xyz_reg1mm.npy`, and `disp_vox_reg1mm_zyx.npy`.
- Smoke runtimes were `204.7 sec` for alpha `0` and `187.9 sec` for alpha
  `1000`, with peak GPU memory `3.96 GB`. These smoke times include full final
  dense-field saving and TRE/regularity evaluation, not only training.
- Passed: initial full launch completed `12/12` jobs with return code 0 on
  GPUs `2,3`.
- Passed: extension launch completed `6/6` new jobs for alphas
  `10000,100000,1000000` with return code 0 on GPUs `2,3`.
- Passed: aggregation completed with `18` rows and no completion errors.

## Results

Primary metric is `original_snap` TRE. All four TRE variants for all 18 rows are
retained in `case_rows.csv` and `review_report.md`.

Boundary/high-alpha rows:

| Case | Alpha | Original snap TRE | Original sub TRE | Reg1mm snap TRE | Reg1mm sub TRE | FG stdLogJ | FG J<=0 % | FG BE fd | FG diffusion fd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 1000 | 0.880 | 0.979 | 1.108 | 0.983 | 0.204629 | 0.0000 | 0.005141 | 0.008681 |
| 01 | 10000 | 0.755 | 0.952 | 1.068 | 0.955 | 0.109595 | 0.0000 | 0.000678 | 0.003496 |
| 01 | 100000 | 0.734 | 0.953 | 1.083 | 0.958 | 0.070804 | 0.0000 | 0.000097 | 0.002076 |
| 01 | 1000000 | 0.815 | 1.007 | 1.113 | 1.013 | 0.052949 | 0.0000 | 0.000016 | 0.001390 |
| 08 | 1000 | 1.048 | 1.141 | 1.261 | 1.136 | 0.354946 | 0.4173 | 0.011372 | 0.035874 |
| 08 | 10000 | 1.087 | 1.147 | 1.271 | 1.144 | 0.192354 | 0.0105 | 0.001444 | 0.013999 |
| 08 | 100000 | 1.110 | 1.201 | 1.320 | 1.208 | 0.128755 | 0.0000 | 0.000270 | 0.010282 |
| 08 | 1000000 | 1.375 | 1.412 | 1.531 | 1.436 | 0.098423 | 0.0000 | 0.000060 | 0.008461 |

Best primary point in the extended coarse grid:

- Case `01`: alpha `100000`, original snap TRE `0.734 mm`, FG stdLogJ
  `0.070804`, FG J<=0 `0.0000%`.
- Case `08`: alpha `1000`, original snap TRE `1.048 mm`, FG stdLogJ
  `0.354946`, FG J<=0 `0.4173%`.

Interpretation: the old alpha `1000` boundary was not high enough for case
`01`; accuracy improved through alpha `100000` and then worsened at `1000000`.
For case `08`, alpha `1000` remains the best original-snap point, while larger
alphas improve regularity and reduce folding but begin to hurt TRE.

## Extension: Larger Alpha Boundary Check

Requested after the initial `0..1000` grid:

```text
additional_alphas = 10000,100000,1000000
cases = 01,08
new_jobs = 6
```

The launcher skipped no extension summaries because these six rows were new,
completed all six jobs, and the final aggregation merged them with the existing
12 baseline-grid rows.
