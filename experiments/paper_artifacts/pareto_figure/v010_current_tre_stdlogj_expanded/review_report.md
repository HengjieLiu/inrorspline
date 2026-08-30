# Current TRE/stdLogJ Expanded Pareto Preview

- Generated: `2026-08-14T23:02:50+00:00`
- OASIS panels are copied from the OASIS100 paper-ready source rows.
- Bottom-left keeps the original LungCT TRE/stdLogJ structure with ellipses.
- Bottom-right is expanded LungCT TRE/stdLogJ without ellipses.
- Masked-loss method IDs are not included in `source_rows.csv`.
- The `w=0.022*` MR-D-BSCP point is tagged as legacy 2400-schedule context.

## Outputs

- `experiments/paper_artifacts/pareto_figure/v010_current_tre_stdlogj_expanded/pareto_four_panel_current_expanded_no_annotations.png`
- `experiments/paper_artifacts/pareto_figure/v010_current_tre_stdlogj_expanded/pareto_four_panel_current_expanded_annotated_review.png`
- `experiments/paper_artifacts/pareto_figure/v010_current_tre_stdlogj_expanded/pareto_four_panel_current_expanded_zoomed_review.png`

## Panel 4 Rows

| Method | Points | TRE range | SDlogJ range |
|---|---:|---:|---:|
| IDIR (IDIR loss) | 7 | 1.006-1.317 | 0.0789-0.3225 |
| MR-D-BSCP (IDIR loss) | 5 | 0.967-1.016 | 0.1181-0.3451 |
| MR-D-BSCP (pTVReg loss) | 6 | 0.917-0.978 | 0.1332-0.3928 |
| Dual-INR PL1A | 6 | 0.976-1.043 | 0.1267-0.3000 |
| INR-BSCP cps4 (FD-BE) | 1 | 1.762-1.762 | 0.2360-0.2360 |
| D-BSCP cps4 (FD-BE) | 1 | 3.894-3.894 | 0.4265-0.4265 |
| pTVReg | 1 | 0.919-0.919 | 0.1977-0.1977 |

## FD-BE Single-Point Case Rows

| Method | Cases | TRE mean | SDlogJ mean | TRE source notes |
|---|---:|---:|---:|---|
| INR-BSCP cps4 (FD-BE) | 10 | 1.762 | 0.2360 | accepted_unmasked_native_tre_bridge, native_crop_tre_snap_to_voxel |
| D-BSCP cps4 (FD-BE) | 10 | 3.894 | 0.4265 | accepted_unmasked_native_tre_bridge, native_crop_tre_snap_to_voxel |
