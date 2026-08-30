# Current TRE/stdLogJ Expanded Pareto Preview with FD-BE cps4 Alpha Landscape

- Generated: `2026-08-15T03:46:29+00:00`
- OASIS panels are copied from the OASIS100 paper-ready source rows.
- Bottom-left keeps the original LungCT TRE/stdLogJ structure with ellipses.
- Bottom-right is expanded LungCT TRE/stdLogJ without ellipses and includes FD-BE cps4 alpha landscapes.
- Alpha=1e4 rerun audit review-only flag: `False`.
- Masked-loss method IDs are not included in `source_rows.csv`.

## Outputs

- `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/pareto_four_panel_current_fdbe_alpha_landscape_no_annotations.png`
- `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/pareto_four_panel_current_fdbe_alpha_landscape_panel4_matched_panel3_y_no_annotations.png`
- `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/pareto_four_panel_current_fdbe_alpha_landscape_annotated_review.png`
- `experiments/paper_artifacts/pareto_figure/v012_current_tre_stdlogj_fdbe_cps4_alpha_landscape/pareto_four_panel_current_fdbe_alpha_landscape_zoomed_review.png`

## Panel 4 Rows

| Method | Points | TRE range | SDlogJ range |
|---|---:|---:|---:|
| IDIR (IDIR loss) | 7 | 1.006-1.317 | 0.0789-0.3225 |
| MR-D-BSCP (IDIR loss) | 5 | 0.967-1.016 | 0.1181-0.3451 |
| MR-D-BSCP (pTVReg loss) | 6 | 0.917-0.978 | 0.1332-0.3928 |
| Dual-INR PL1A | 6 | 0.976-1.043 | 0.1267-0.3000 |
| INR-BSCP cps4 (FD-BE) | 7 | 1.042-1.762 | 0.0858-0.2360 |
| D-BSCP cps4 (FD-BE) | 7 | 1.016-3.894 | 0.0857-0.4265 |
| pTVReg | 1 | 0.919-0.919 | 0.1977-0.1977 |

## FD-BE cps4 Alpha-Landscape Case Rows

| Method | Alpha | Cases | TRE mean | SDlogJ mean | Source rows |
|---|---:|---:|---:|---:|---|
| D-BSCP cps4 (FD-BE) | e4 | 10 | 3.894 | 0.4265 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | 2e4 | 10 | 2.743 | 0.2854 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | 5e4 | 10 | 1.306 | 0.1592 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | e5 | 10 | 1.016 | 0.1182 | experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | 2e5 | 10 | 1.057 | 0.1065 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | 5e5 | 10 | 1.144 | 0.0940 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| D-BSCP cps4 (FD-BE) | e6 | 10 | 1.209 | 0.0857 | experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | e4 | 10 | 1.762 | 0.2360 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | 2e4 | 10 | 1.474 | 0.1795 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | 5e4 | 10 | 1.042 | 0.1351 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | e5 | 10 | 1.043 | 0.1193 | experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | 2e5 | 10 | 1.069 | 0.1068 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | 5e5 | 10 | 1.156 | 0.0940 | experiments/2026-08-14_lungct_fdbe_cps4_alpha_landscape_all10/case_rows.csv |
| INR-BSCP cps4 (FD-BE) | e6 | 10 | 1.218 | 0.0858 | experiments/2026-08-14_lungct_fdbe_cps4_alpha1e5_1e6_all10/case_rows.csv |
