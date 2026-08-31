# Technical log

## Known limitations

- This is a visual and geometry audit, not a segmentation-accuracy evaluation.
- R231 is used only to define the shared coronal extent and as one displayed
  segmentation source. The other two rows retain their supplied lobe labels.
- A shared R231 pair-union range improves phase-to-phase visual comparison but
  does not make the two respiratory phases anatomically aligned.
- The figures use a CT lung window and opacity overlays; subtle boundary errors
  may require an additional focused review.
- No figure files are saved by the notebook.

## Execution record

- Created: 2026-08-30.
- Intended environment: `py310sitreg_v1_test1`.
- Dataset: Learn2Breath challenge-like COPD10 pair-local product, 224 x 192 x
  224 at 1.5 mm.
- Requested figures: all cases 104--113, three segmentation rows, and paired
  INSP/EXP panels at 25%, 50%, and 75% of the R231-derived coronal extent.
- Executed successfully in the intended Docker container on 2026-08-30.
- Validation: all 20 image/phase grids, 60 displayed source masks, and 40
  TotalSegmentator lobe-union companions passed shape, affine, spacing,
  non-emptiness, label-set, and union-consistency checks.
- Rendered review: 10 inline three-row by six-column figures; no external
  figure files or data products were created.
