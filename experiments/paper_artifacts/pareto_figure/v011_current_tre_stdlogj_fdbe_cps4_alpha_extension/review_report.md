# Current TRE/stdLogJ Expanded Pareto Preview with FD-BE cps4 Alpha Extension

- Generated: `2026-08-15T00:18:11+00:00`
- OASIS panels are copied from the OASIS100 paper-ready source rows.
- Bottom-left keeps the original LungCT TRE/stdLogJ structure with ellipses.
- Bottom-right is expanded LungCT TRE/stdLogJ without ellipses and includes FD-BE cps4 alpha sweeps.
- Masked-loss method IDs are not included in `source_rows.csv`.
- The `w=0.022*` MR-D-BSCP point is tagged as legacy 2400-schedule context.

## Outputs

- `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/pareto_four_panel_current_fdbe_alpha_extension_no_annotations.png`
- `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/pareto_four_panel_current_fdbe_alpha_extension_annotated_review.png`
- `experiments/paper_artifacts/pareto_figure/v011_current_tre_stdlogj_fdbe_cps4_alpha_extension/pareto_four_panel_current_fdbe_alpha_extension_zoomed_review.png`

## Panel 4 Rows

| Method | Points | TRE range | SDlogJ range |
|---|---:|---:|---:|
| IDIR (IDIR loss) | 7 | 1.006-1.317 | 0.0789-0.3225 |
| MR-D-BSCP (IDIR loss) | 5 | 0.967-1.016 | 0.1181-0.3451 |
| MR-D-BSCP (pTVReg loss) | 6 | 0.917-0.978 | 0.1332-0.3928 |
| Dual-INR PL1A | 6 | 0.976-1.043 | 0.1267-0.3000 |
| INR-BSCP cps4 (FD-BE) | 3 | 1.043-1.762 | 0.0858-0.2360 |
| D-BSCP cps4 (FD-BE) | 3 | 1.016-3.894 | 0.0857-0.4265 |
| pTVReg | 1 | 0.919-0.919 | 0.1977-0.1977 |

## FD-BE cps4 Alpha-Sweep Case Rows

| Method | Alpha | Cases | TRE mean | SDlogJ mean | TRE source notes |
|---|---:|---:|---:|---:|---|
| D-BSCP cps4 (FD-BE) | 10000 | 10 | 3.894 | 0.4265 | accepted_unmasked_native_tre_bridge, native_crop_tre_snap_to_voxel |
| D-BSCP cps4 (FD-BE) | 100000 | 10 | 1.016 | 0.1182 | native_crop_tre_snap_to_voxel |
| D-BSCP cps4 (FD-BE) | 1e+06 | 10 | 1.209 | 0.0857 | native_crop_tre_snap_to_voxel |
| INR-BSCP cps4 (FD-BE) | 10000 | 10 | 1.762 | 0.2360 | accepted_unmasked_native_tre_bridge, native_crop_tre_snap_to_voxel |
| INR-BSCP cps4 (FD-BE) | 100000 | 10 | 1.043 | 0.1193 | native_crop_tre_snap_to_voxel |
| INR-BSCP cps4 (FD-BE) | 1e+06 | 10 | 1.218 | 0.0858 | native_crop_tre_snap_to_voxel |
