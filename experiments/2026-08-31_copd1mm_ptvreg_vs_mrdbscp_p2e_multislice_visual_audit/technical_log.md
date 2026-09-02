# Technical log

## Known limitations

- This audit reconstructs existing fields; it does not rerun either method.
- MATLAB pTVReg and P2e did not save fields on identical registration grids.
  pTVReg is brought to the P2e canonical grid through its saved native full-grid
  physical pull field rather than through a second image resize.
- Canonical fixed and moving lung masks are used only for AP-slice selection and
  visual contours. They are not retroactively added to either objective.
- Per-case 99th-percentile display limits improve within-case inspection. Values
  are printed in the figure title and should not be treated as cohort-wide
  normalization.

## Acceptance conditions

- All ten cases load finite CT, mask, and field arrays on the expected grids.
- pTVReg landmark and native-deformed-volume contracts, plus the P2e landmark
  and normalized-field contracts, pass before rendering clinical images.
- The executed notebook embeds ten five-by-eight case figures with the stated
  SI/LR orientation markers and AP slice metadata.
