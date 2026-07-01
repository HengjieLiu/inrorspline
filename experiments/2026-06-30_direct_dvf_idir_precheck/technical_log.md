# Direct Dense-DVF vs IDIR Precheck Technical Log

## Known Limitations

- Direct dense-grid DVF is not continuous between grid points, so native
  analytic IDIR-style bending energy is unavailable.
- OASIS and LungCT direct DVF both use finite differences on the dense grid for
  BE. Values are comparable as dense-grid diagnostics, not as the same analytic
  object as IDIR autograd BE.
- LungCT cases use the canonical pTV-preprocessed cropped 1mm diagnostic
  domain. These rows should not be interpreted as native/full-volume pTVReg
  headline comparisons.
- Direct DVF image samples are fixed-domain grid points. Unlike IDIR/SIREN,
  displacement values are not queried continuously.

## Launch Notes

- Planned GPUs: OASIS rank9 on `0`, LungCT case01 on `2`, LungCT case08 on `3`.
- LR: `1e-4`.
- Epochs: `2500`.
- Seed: `1`.
- Samples per epoch: `10000`.
- OASIS alpha: `1`.
- LungCT alpha_bending: `10`.

## Status

- Completed.

## Preflight

- Compile/import checks passed inside Docker for modified OASIS and LungCT
  runners, LungCT common utilities, and the aggregation script.
- OASIS rank9 validation passed: original pair index `88`, fixed `0173`,
  moving `0136`.
- LungCT loader audit passed for cases `01,08` with preprocessing verification.
- Two-epoch smokes passed for OASIS rank9 and LungCT case01 direct dense-DVF.

## Full Launch

- OASIS rank9 ran on GPU `0`.
- LungCT case01 ran on GPU `2`.
- LungCT case08 ran on GPU `3`.
- GPU `1` was not used.
- All `3/3` jobs completed.

## Result Snapshot

| Dataset | Case | Direct Dense-DVF | IDIR Anchor |
| --- | --- | ---: | ---: |
| OASIS | rank9 Dice mean | 0.7673 | 0.8110 |
| OASIS | rank9 foreground folding % | 9.889 | 1.560 |
| LungCT | case01 snap TRE mm | 3.764 | 1.063 |
| LungCT | case08 snap TRE mm | 15.076 | 1.331 |

Direct dense-DVF was feasible, but at LR `1e-4` and the matched alpha anchors it
was worse than IDIR on accuracy/TRE and produced much rougher fields. Treat this
as a negative preliminary check for a broad same-settings expansion, not as a
tuned dense-DVF result.

## Review Artifacts

- `review_report.md`
- `direct_dvf_idir_precheck_rows.csv`
- `direct_dvf_idir_precheck_rows.json`
