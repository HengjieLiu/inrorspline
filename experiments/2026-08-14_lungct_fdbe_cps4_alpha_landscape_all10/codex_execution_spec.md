# Codex Execution Spec: LungCT FD-BE cps4 Alpha Landscape

## Objective

Run and aggregate the unmasked LungCT FD-BE cps4 alpha landscape for INR-BSCP
and D-BSCP on DIR-Lab 4DCT cases 01-10.

## Inputs

- Data: canonical pTVReg iso1mm DIR-Lab 4DCT cases `01-10`.
- Methods:
  - INR-BSCP: `sinr_cps4_global_ncc_fd_be`, `lr=1e-4`.
  - D-BSCP: `direct_cp_cps4_global_ncc_fd_be`, `lr=1e-3`.
- Shared settings: `cps=4`, `epochs=2500`, `seed=1`, full-crop global NCC,
  dense FD-BE regularization, fixed T00 lung-mask metric diagnostics only.

## Launch Plan

- Output root:
  `/database/inrorspline/runs/lungct_fdbe_cps4_alpha_landscape_all10_v1`.
- Wave 1: alpha lanes `2e4`, `5e4`, `2e5`, `5e5` on GPUs `0,1,2,3`.
  Each lane runs INR-BSCP cases `01-10`, then D-BSCP cases `01-10`.
- Wave 2: alpha `1e4` rerun for homogeneous diagnostics.
  INR-BSCP uses GPU `0`; D-BSCP uses GPU `1`.
- Existing accepted rows for `1e5` and `1e6` are reused from
  `experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/case_rows.csv`.

## Acceptance Gates

- Dry-run Wave 1 reports exactly `80` jobs.
- Dry-run Wave 2 reports exactly `20` jobs.
- Smoke Wave 1 and Wave 2 complete with `--epoch-override 1`.
- Full launch completes `100/100` current-source rows.
- Aggregation finds no masked-loss method IDs, protocol tags, or source paths.
- Current-source rows have finite TRE, SDlogJ, folding, full-crop `1-NCC`,
  fixed-mask diagnostic `1-NCC`, runtime, peak memory, final data loss, and
  final regularization loss.
- v012 combined FD-BE case table has `140` rows.
- Alpha `1e4` rerun-vs-v011 audit is written; if mean TRE differs by more than
  `0.05 mm` or any case differs by more than `0.25 mm`, v012 is marked
  review-only in its manifest.
