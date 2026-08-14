# LungCT TRE/stdLogJ Existing-Run Design Review

- Generated: `2026-08-12T21:54:10+00:00`
- No IDIR pTVReg-loss sweep was launched or required for this preview.
- A2 is the all-candidate no-ellipse panel.
- A2, B1, and B2 use A1's y-axis limits inside each 2x2 figure.
- B1 and B2 also use matched x-axis limits inside each 2x2 figure.
- Ellipse width/height are `2 * sample SD` for foreground stdLogJ and original snap TRE.
- The IDIR pTV pilot is annotated-only and is not included in the unannotated camera-style preview.

## Outputs

- `experiments/paper_artifacts/pareto_figure/v008_lungct_existing_run_design_review/lungct_tre_stdlogj_2x2_annotated.png`
- `experiments/paper_artifacts/pareto_figure/v008_lungct_existing_run_design_review/lungct_tre_stdlogj_2x2_unannotated.png`

## Plotted Unannotated Rows

| Method | Points | TRE range | stdLogJ range |
|---|---:|---:|---:|
| IDIR (IDIR loss) | 7 | 1.006-1.317 | 0.0789-0.3225 |
| MR-D-BSCP (IDIR loss) | 5 | 0.966-1.017 | 0.1181-0.3475 |
| MR-D-BSCP (pTVReg loss) | 4 | 0.922-1.022 | 0.1332-0.3974 |
| Dual-INR (original DUAL loss) | 6 | 0.976-1.043 | 0.1267-0.3000 |
| Dual-INR (pTVReg loss) | 5 | 0.971-1.007 | 0.1102-0.3047 |
| pTVReg (pTVReg loss) | 1 | 0.919-0.919 | 0.1977-0.1977 |

## Exclusions

- Base MR-D-BSCP/IDIR curves use the existing LungCT default-exclusion convention.
- DUAL PL1A `m100` is omitted from the preview.
- Full included/excluded row metadata is retained in `source_rows.csv`.
