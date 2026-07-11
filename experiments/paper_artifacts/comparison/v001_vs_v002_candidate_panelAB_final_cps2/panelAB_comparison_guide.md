# Panel A/B Final-CPS Matched Comparison Guide

Purpose: compare v001 against v002_candidate after switching OASIS Panel A/B single-stage spline methods from cps4 to cps2.

Primary paper-layout comparison:
- Old: `experiments/paper_artifacts/fitting_figure/v001/panelAB_composite/fit_modes_panelAB_composite_sitreg.png`
- New: `experiments/paper_artifacts/fitting_figure/v002_candidate/panelAB_composite/fit_modes_panelAB_composite_sitreg.png`

Standard generator comparison:
- Old: `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_panelAB_curves.png`
- New: `experiments/paper_artifacts/fitting_figure/v002_candidate/fit_modes_standardized_panelAB_curves.png`

Side-by-side review image:
- `experiments/paper_artifacts/comparison/v001_vs_v002_candidate_panelAB_final_cps2/panelAB_v001_vs_v002_side_by_side.png`

What changed:
- OASIS INR-BSCP uses cps2 in v002_candidate instead of cps4 in v001.
- OASIS D-BSCP uses cps2 in v002_candidate instead of cps4 in v001.
- OASIS MR-D-BSCP stays the same 8-4-2 run.
- LungCT selections stay unchanged: single-stage cps4 and multistage 32-16-8-4.

Audit counts:
- Source-changed rows total: 4.
- Source-changed OASIS rows: 4.
- Source-changed LungCT rows: 0.

Interpretation:
v002_candidate is the final-cps matched OASIS Panel A/B review candidate. It should be compared against v001 mainly in the OASIS column; LungCT panels should be visually and numerically unchanged except for any regeneration-level formatting differences.
