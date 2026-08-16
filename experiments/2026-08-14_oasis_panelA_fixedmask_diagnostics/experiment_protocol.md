# OASIS Panel A Fixed-Mask Diagnostic Refresh Protocol

## Motivation

This v003 refresh fixes the last metric-domain inconsistency in the population
diagnostic figures.  v002 already evaluates DIR-LAB Panel A inside the fixed
T00 lung mask.  This experiment refreshes OASIS Panel A so the plotted
registration diagnostic is also evaluated in the dataset's fixed anatomical
foreground domain: fixed raw `image > 0`.

The optimization objectives are unchanged.  This is a diagnostic/evaluation
rerun, not a new training protocol.

## Cohort and Methods

- Dataset: OASIS BrainMR seed-3 shuffled `pair_test_200` first10.
- Direction: moving image to fixed image as written in
  `configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`.
- Cases: ranks `0-9`.
- Methods:
  - INR-Dense: `idir_original`, alpha `1`, LR `1e-4`, `2500` epochs.
  - INR-BSCP: `sinr_cps2_be_bspline_analytic`, alpha `1000`, LR `1e-4`,
    `2500` epochs.
  - D-BSCP: `direct_cp_cps2_be_bspline_analytic`, alpha `1000`, LR `1e-3`,
    `2500` epochs.
  - MR-D-BSCP:
    `direct_cp_8_4_2_be_bspline_analytic_multistage`, alpha `1000`, LR
    `1e-3`, cps `8/4/2`, downsample `4/2/1`, stages `400/800/1300`.

## Diagnostic Metrics

Each OASIS Panel A run writes paired NCC diagnostics every 10 epochs, including
epoch 1 and epoch 2500:

- Full-image global `1-NCC`, retained as a reproducibility cross-check.
- Fixed raw `image > 0` masked `1-NCC`, used as the v003 plotted metric.

The NCC formula is unchanged:

```text
ncc_loss = -corr(fixed, warped)
1-NCC    = 1 + ncc_loss
```

## GPU Schedule

Use four GPUs with fixed paired shards:

- GPU 0: ranks `0-4`, INR-BSCP then D-BSCP.
- GPU 1: ranks `5-9`, INR-BSCP then D-BSCP.
- GPU 2: ranks `0-4`, INR-Dense then MR-D-BSCP.
- GPU 3: ranks `5-9`, INR-Dense then MR-D-BSCP.

This pairs the slowest and fastest methods on GPUs `0/1`, and the two
middle-speed methods on GPUs `2/3`.

## Outputs

Large optimization outputs are written outside Git under the portable token
`${OASIS_MASKED_PANELA_RUN_ROOT}`.  Compact manifests, validation reports, and
figure artifacts are written under:

- `experiments/2026-08-14_oasis_panelA_fixedmask_diagnostics/`
- `experiments/paper_artifacts/fit_modes_population_diagnostics/v003_masked_panelA_both_datasets/`

v003 preserves all non-OASIS-Panel-A rows from v002 and replaces exactly 40
OASIS Panel A records.

## Acceptance

- Dry-run resolves exactly 40 jobs.
- Smoke rank 0 runs all four methods with fixed-mask diagnostic columns.
- Full run completes `40/40` records.
- Every full curve has 251 finite positive fixed-mask `1-NCC` points ending at
  epoch 2500.
- v003 source index has exactly 160 records and exactly 40 replacements.
- DIR-LAB v002 Panel A and all Panel B rows remain unchanged.
