# LungCT DUAL-INR PL1A/PL1C TRE-SDlogJ Overlay

- Generated: `2026-08-11T23:34:26+00:00`
- Base source rows: `experiments/paper_artifacts/pareto_figure/v002_candidate/lungct10_pareto_source_rows.csv`
- DUAL source rows: `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage2_curve_points.csv`
- Combined source rows: `experiments/paper_artifacts/pareto_figure/v004_candidate_dual_inr_pl1a_pl1c/lungct10_dual_overlay_source_rows.csv`
- Ellipses show +/-1 sample SD in both axes via full width/height `2 * SD`.

## Outputs

- `experiments/paper_artifacts/pareto_figure/v004_candidate_dual_inr_pl1a_pl1c/lungct10_dual_overlay/default_exclusions/original_snap_tre_mean_mm_vs_fg_logj_std.png`
- `experiments/paper_artifacts/pareto_figure/v004_candidate_dual_inr_pl1a_pl1c/lungct10_dual_overlay_paper_no_annotations/default_exclusions/original_snap_tre_mean_mm_vs_fg_logj_std.png`

## Added DUAL-INR Points

| Method | Weight | n | TRE mean +/- SD | stdLogJ mean +/- SD |
|---|---:|---:|---:|---:|
| dual_inr_pl1a_author_blur_native_loss | m=0.01 | 10 | 1.043 +/- 0.178 | 0.3000 +/- 0.0972 |
| dual_inr_pl1a_author_blur_native_loss | m=0.1 | 10 | 0.991 +/- 0.166 | 0.2017 +/- 0.0564 |
| dual_inr_pl1a_author_blur_native_loss | m=1 | 10 | 0.976 +/- 0.175 | 0.1434 +/- 0.0374 |
| dual_inr_pl1a_author_blur_native_loss | m=10 | 10 | 1.040 +/- 0.202 | 0.1267 +/- 0.0344 |
| dual_inr_pl1a_author_blur_native_loss | m=100 | 10 | 1.137 +/- 0.253 | 0.1303 +/- 0.0372 |
| dual_inr_pl1c_author_blur_ptv_loss | w=0.0055 | 10 | 1.007 +/- 0.170 | 0.3047 +/- 0.0804 |
| dual_inr_pl1c_author_blur_ptv_loss | w=0.017 | 10 | 0.986 +/- 0.161 | 0.2697 +/- 0.0723 |
| dual_inr_pl1c_author_blur_ptv_loss | w=0.055 | 10 | 0.991 +/- 0.173 | 0.2099 +/- 0.0599 |
| dual_inr_pl1c_author_blur_ptv_loss | w=0.17 | 10 | 0.971 +/- 0.170 | 0.1513 +/- 0.0457 |
| dual_inr_pl1c_author_blur_ptv_loss | w=0.55 | 10 | 0.980 +/- 0.198 | 0.1102 +/- 0.0348 |
