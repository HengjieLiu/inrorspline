# Panel A/B v001 vs v002_candidate Delta Summary

v002_candidate changes the OASIS single-stage spline Panel A/B selections to final-cps matched cps2 variants. v001 is unchanged.

Protocol changes:
- OASIS INR-BSCP: cps4 -> cps2.
- OASIS D-BSCP: cps4 -> cps2.
- OASIS MR-D-BSCP: unchanged 8-4-2.
- LungCT: unchanged cps4 single-stage and 32-16-8-4 multistage.

Endpoint changes for OASIS:

| Panel | Method | Endpoint | v001 | v002_candidate | Delta | Delta % |
|---|---|---:|---:|---:|---:|---:|
| A | D-BSCP | final 1-NCC | 0.00889 | 0.00710 | -0.00178 | -20.0% |
| A | INR-BSCP | final 1-NCC | 0.00898 | 0.00830 | -0.00069 | -7.7% |
| A | MR-D-BSCP | final 1-NCC | 0.00597 | 0.00597 | 0.00000 | 0.0% |
| B | D-BSCP | final rRMSE (%) | 11.799 | 4.474 | -7.325 | -62.1% |
| B | INR-BSCP | final rRMSE (%) | 12.976 | 11.417 | -1.559 | -12.0% |
| B | MR-D-BSCP | final rRMSE (%) | 4.503 | 4.503 | 0.00000 | 0.0% |

LungCT source-change check:
- Changed LungCT source rows: 0.

Files to compare:
- Old paper-layout figure: `experiments/paper_artifacts/fitting_figure/v001/panelAB_composite/fit_modes_panelAB_composite_sitreg.png`
- New paper-layout figure: `experiments/paper_artifacts/fitting_figure/v002_candidate/panelAB_composite/fit_modes_panelAB_composite_sitreg.png`
- Old standard generator figure: `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_panelAB_curves.png`
- New standard generator figure: `experiments/paper_artifacts/fitting_figure/v002_candidate/fit_modes_standardized_panelAB_curves.png`
