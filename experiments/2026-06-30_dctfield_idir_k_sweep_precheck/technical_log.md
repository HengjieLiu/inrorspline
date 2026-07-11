# DCTField IDIR-Loss K-Sweep Precheck Technical Log

## Known Limitations

- DCTField is decoded only on grid points in this first round; continuous
  off-grid DCT evaluation is not implemented.
- Training BE is dense finite difference on the decoded DCT field. Spectral DCT
  analytic BE is possible later but not used here.
- LungCT rows use the canonical pTV-preprocessed cropped 1mm diagnostic domain,
  not native/full-volume headline TRE.
- This is a K-size precheck at fixed LR/alpha, not a tuned DCTField result.

## Launch Notes

- Planned GPU queues: OASIS on `0`, LungCT case01 on `2`, LungCT case08 on `3`.
- GPU `1` must not be used.
- K values: `16,32,64,128`.
- LR: `1e-4`.
- Epochs: `2500`.
- Seed: `1`.
- Samples per epoch: `10000`.
- OASIS alpha: `1`.
- LungCT alpha_bending: `10`.

## Status

- Completed.

## Preflight

- Compile/import checks passed inside Docker for the shared DCT helper,
  modified runners, DCT sanity script, and aggregation script.
- DCT decoder sanity passed: zero coefficients decode to zero, decoded shapes
  match requested image shapes, gradients flow to coefficients, and oversized K
  raises a clear error.
- OASIS rank9 validation passed: original pair index `88`, fixed `0173`,
  moving `0136`.
- LungCT loader audit passed for cases `01,08` with preprocessing verification.
- Two-epoch K128 smokes passed for OASIS rank9 and LungCT case08 without OOM or
  NaN.

## Full Launch

- OASIS K `16,32,64,128` ran sequentially on GPU `0`.
- LungCT case01 K `16,32,64,128` ran sequentially on GPU `2`.
- LungCT case08 K `16,32,64,128` ran sequentially on GPU `3`.
- GPU `1` was not used.
- All `12/12` jobs completed.

## Result Snapshot

| Dataset | Case | Best DCTField | IDIR Anchor | Direct Dense-DVF |
| --- | --- | ---: | ---: | ---: |
| OASIS | rank9 Dice mean | K128 0.7781 | 0.8110 | 0.7673 |
| OASIS | rank9 foreground folding % | K128 16.89 | 1.560 | 9.889 |
| LungCT | case01 snap TRE mm | K32 3.652 | 1.063 | 3.764 |
| LungCT | case08 snap TRE mm | K64 14.986 | 1.331 | 15.076 |

OASIS K32 is the smoother DCT tradeoff: Dice `0.7242` with foreground folding
`0.265%`. Larger DCT bases improve OASIS overlap but reintroduce folding.
LungCT case01 gets a small smoothness/accuracy improvement over direct dense
DVF at K32. LungCT case08 remains essentially unrecovered under this
single-stage fixed LR/alpha DCT setup.

## Review Artifacts

- `review_report.md`
- `dctfield_idir_k_sweep_rows.csv`
- `dctfield_idir_k_sweep_rows.json`
