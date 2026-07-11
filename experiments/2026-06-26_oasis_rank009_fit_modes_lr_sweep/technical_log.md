# OASIS Rank-9 Fit-Mode LR Sweep Technical Log

## Status

- Planned on 2026-06-26.
- Implementation target: repo-local OASIS alpha-sweep utilities.
- Docker target: `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.

## Known Limitations Before Launch

- This is a single-case diagnostic using OASIS seed-3 rank `9`
  `(0173,0136)`, selected because canonical Greedy has the best Dice among the
  seed-3 first10 Greedy rows.
- IDIR alpha is fixed to `1`, the completed first10 sweep's best IDIR alpha by
  population mean Dice, HD95, and symmetric ASD. IDIR alpha `0.1` is not
  included in this run.
- IDIR BE scaling is not changed during training. Any IDIR scale correction is
  a plot-only diagnostic derived from the saved history.
- Direct Greedy-field fitting uses normalized-grid MSE over the fixed-image-
  positive brain mask. It does not use image loss, label metrics, or
  regularization during training.
- The brain mask is the raw fixed OASIS image after standard reorientation with
  `fixed_raw > 0`. Segmentation foreground, `seg > 0`, is reserved for
  Dice/HD95/ASD and visual audits.
- IDIR direct field fitting samples `10,000` fixed-image-positive coordinates
  per epoch, matching the IDIR registration optimization style. SINR/DirectCP
  expand their cps4 dense field each epoch and compute MSE on all fixed-image-
  positive voxels.

## Acceptance Checks Planned

- Exactly `18` completed summaries: 2 objectives times 3 methods times 3
  learning rates.
- Every row resolves to case rank `9`, original sorted index `88`, fixed/moving
  `(0173,0136)`.
- Greedy target audit reports the canonical saved-baseline orientation
  convention: 5D channel-last, scalar LIA-to-RAS per component, signs `-,-,+`.
- Image-registration rows use original runner semantics: IDIR fixed-image
  foreground coordinate sampling; SINR/DirectCP full-image NCC; method-native
  BE regularization.
- Field-fit rows use MSE-only training on the fixed-image-positive brain mask
  and normalized-grid target units. Full-volume MSE is retained as a posthoc
  context metric, not the training objective.
- Every run has `DONE`, `summary.json`, `history.csv`, `case_metrics.csv`,
  `regularity_metrics.csv`, `be_metrics.csv`, `diffusion_metrics.csv`,
  `model_state.pt`, `mask_audit.json`, and final dense displacement output.
