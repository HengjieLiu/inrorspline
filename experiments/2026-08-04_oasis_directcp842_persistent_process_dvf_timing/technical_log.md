# Technical Log: Persistent-Process DirectCP + SINR DVF Timing

## Known Limitations

- Runtime-only characterization; no Dice, HD95/ASD, dense NCC, or regularity
  metrics are computed.
- Warm ranks `1-9` are different OASIS pairs from rank `0`, so
  `cold_minus_warm_mean` is an estimate of first-use overhead plus case/runtime
  differences, not a pure hardware constant.
- Original and `A_fast_decay` are run in separate persistent worker
  subprocesses to avoid order bias from shared CUDA/cuDNN state.
- SINR cps4 and SINR cps2 are added as an extension in separate persistent
  worker subprocesses. They reuse the same DVF-only timing definitions but have
  one fixed-cps stage and no knot insertion/prolongation.

## Launch Notes

- Date: 2026-08-04.
- Container: `py310sitreg_v1_test1`.
- GPU policy: serial worker subprocesses on GPU `0`; GPUs `1-3` intended idle.
- Production output root:
  `<external-run-root>/runs/oasis_directcp842_persistent_dvf_timing_alpha1000_v1`.

## Results Log

- 2026-08-04 extension planned: add `sinr_cps4_be_bspline_analytic` and
  `sinr_cps2_be_bspline_analytic` at alpha `1000`, lr `1e-4`, full `2500`
  epochs, seed `1`, preserving existing DirectCP rows and regenerating the
  same report.
- Smoke completed for both methods on ranks `0,1` with `1,1,1` epochs.
  Original cold rank0 wall was `130.68 s`, warm rank1 wall was `1.79 s`;
  `A_fast_decay` cold rank0 wall was `109.33 s`, warm rank1 wall was
  `3.20 s`.
- Full first10 completed with one persistent worker subprocess per candidate.
  DirectCP P22 rows were reused, and SINR cps4/cps2 added `20/20` new case
  rows. The combined report has `40/40` case rows with dense DVF,
  `summary.json`, `timing.json`, and `DONE` files.
- Evaluation/full-output events counted in DVF-only runs: `0`.
- Reconciliation residual failures: `0`.
- Parent self-unaccounted warnings: `0`.
- Non-target GPU compute-app observations: `0`.

## Full First10 Timing

| Method | Group | Cases | Full script -> DVF memory | Loaded images -> DVF memory | Full script -> DVF file | Wall/case | Epochs |
|---|---|---:|---:|---:|---:|---:|---:|
| Original MR-D-BSCP / DirectCP `8-4-2` | cold rank0 | 1 | 177.41 s | 147.71 s | 177.48 s | 177.49 s | 2500 |
| Original MR-D-BSCP / DirectCP `8-4-2` | warm ranks1-9 | 9 | 62.48 +/- 0.17 s | 62.07 +/- 0.19 s | 62.55 +/- 0.17 s | 62.56 +/- 0.17 s | 2500 |
| Original MR-D-BSCP / DirectCP `8-4-2` | amortized ranks0-9 | 10 | 73.97 +/- 36.35 s | 70.63 +/- 27.08 s | 74.05 +/- 36.34 s | 74.06 +/- 36.34 s | 2500 |
| `A_fast_decay` | cold rank0 | 1 | 144.55 s | 116.93 s | 144.63 s | 144.64 s | 600 |
| `A_fast_decay` | warm ranks1-9 | 9 | 15.07 +/- 0.69 s | 14.52 +/- 0.65 s | 15.18 +/- 0.69 s | 15.19 +/- 0.69 s | 575 +/- 21.7 |
| `A_fast_decay` | amortized ranks0-9 | 10 | 28.02 +/- 40.95 s | 24.76 +/- 32.39 s | 28.12 +/- 40.94 s | 28.14 +/- 40.94 s | 577.5 +/- 21.9 |
| SINR cps4 | cold rank0 | 1 | 182.49 s | 155.60 s | 182.81 s | 182.81 s | 2500 |
| SINR cps4 | warm ranks1-9 | 9 | 107.39 +/- 0.09 s | 106.97 +/- 0.09 s | 107.48 +/- 0.09 s | 107.49 +/- 0.09 s | 2500 |
| SINR cps4 | amortized ranks0-9 | 10 | 114.90 +/- 23.75 s | 111.83 +/- 15.38 s | 115.02 +/- 23.82 s | 115.02 +/- 23.82 s | 2500 |
| SINR cps2 | cold rank0 | 1 | 356.39 s | 327.62 s | 356.53 s | 356.54 s | 2500 |
| SINR cps2 | warm ranks1-9 | 9 | 271.38 +/- 0.15 s | 270.92 +/- 0.16 s | 271.45 +/- 0.15 s | 271.45 +/- 0.15 s | 2500 |
| SINR cps2 | amortized ranks0-9 | 10 | 279.88 +/- 26.89 s | 276.59 +/- 17.93 s | 279.95 +/- 26.91 s | 279.96 +/- 26.91 s | 2500 |

Warm persistent-process savings for `A_fast_decay` relative to original were
`47.40 s` to in-memory DVF, `47.37 s` to DVF file, and `47.37 s` wall per case.
Amortized first10 savings were `45.95 s` to in-memory DVF and `45.92 s` wall
per case.

Warm persistent-process SINR comparison: SINR cps4 was `44.93 s` slower than
original DirectCP `8-4-2` and `92.30 s` slower than `A_fast_decay` by external
wall. SINR cps2 was `208.89 s` slower than original DirectCP `8-4-2` and
`256.26 s` slower than `A_fast_decay` by external wall.

## Primary Artifacts

- `persistent_process_timing_report.md`
- `persistent_process_summary.csv`
- `persistent_process_case_timing.csv`
- `persistent_process_savings.csv`
- `persistent_process_pairwise_runtime_comparison.csv`
- `persistent_process_cold_warm_breakdown.csv`
- `dvf_only_method_comparison.csv`
- `dvf_only_stage_breakdown.csv`
- `command_timing_dvf_only_first10.csv`
