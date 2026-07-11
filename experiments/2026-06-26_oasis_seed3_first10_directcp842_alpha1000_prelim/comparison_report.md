# OASIS DirectCP 8-4-2 Alpha1000 Preliminary Comparison

- Generated: `2026-06-26 17:49:28 +0000`
- New output root: `/database/inrorspline/runs/oasis_seed3_first10_directcp842_alpha1000_prelim_v1`
- Review root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_seed3_first10_directcp842_alpha1000_prelim`
- Reference root: `/homebase/code_sync/inrorspline/experiments/2026-06-25_oasis_seed3_first10_alpha_sweep/direct_cp_cps2_full_overlay`
- Next10 reference root: `/homebase/code_sync/inrorspline/experiments/2026-06-26_oasis_seed3_next10_alpha_sweep`
- New summaries: `20` / `20`
- New case ranks: `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19`
- New method: DirectCP multistage `8-4-2`, downsample `4-2-1`, stage epochs `400-800-1300`, lr `1e-2`, alpha `1000`.

## Acceptance

- PASS: all new rows and required artifacts are present with expected metadata.

## Comparison Tables

### first10

- Best mean Dice: `DirectCP 8-4-2 alpha1000` with Dice `0.8054`.
- New mean Dice/HD95/symmetric ASD: `0.8054` / `2.2429` / `0.4507`.
- New foreground native BE/logJ std/fold%: `0.1137` / `0.8826` / `4.3015`.

| Method | n | alpha | lr | dice mean | hd95 mean | asd warped to fixed mean | asd fixed to warped mean | asd symmetric mean | be fd forward fixed fg voxel | be native analytic fixed fg voxel | diffusion l2 fg | diffusion fd forward fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR best Dice | 10 | 1 | 1.000e-04 | 0.7721 | 2.3415 | 0.5594 | 0.4591 | 0.5300 | 0.0306 | 0.0310 | 0.1047 | 0.1047 | 0.1050 | 0.7277 | 2.2658 | 0.0061 |
| SINR cps4 alpha1000 | 10 | 1e+03 | 1.000e-04 | 0.7945 | 2.2745 | 0.5229 | 0.3788 | 0.4695 | 0.0895 | 0.0942 | 0.1857 | 0.1857 | 0.1867 | 0.9369 | 5.6784 | 0.0198 |
| SINR cps2 alpha1000 | 10 | 1e+03 | 1.000e-04 | 0.7992 | 2.2374 | 0.5048 | 0.3664 | 0.4535 | 0.0951 | 0.1007 | 0.1806 | 0.1806 | 0.1816 | 0.9237 | 5.4108 | 0.0194 |
| DirectCP cps4 alpha1000 | 10 | 1e+03 | 0.00100 | 0.7965 | 2.3120 | 0.5243 | 0.3707 | 0.4672 | 0.1116 | 0.1185 | 0.1847 | 0.1847 | 0.1859 | 0.9191 | 5.7186 | 0.0227 |
| DirectCP cps2 alpha1000 | 10 | 1e+03 | 0.00100 | 0.8023 | 2.2788 | 0.5272 | 0.3562 | 0.4637 | 0.1274 | 0.1420 | 0.1769 | 0.1769 | 0.1784 | 0.9118 | 5.7552 | 0.0255 |
| DirectCP 8-4-2 alpha1000 | 10 | 1e+03 | 0.01000 | 0.8054 | 2.2429 | 0.5056 | 0.3525 | 0.4507 | 0.1019 | 0.1137 | 0.1627 | 0.1627 | 0.1639 | 0.8826 | 4.3015 | 0.0147 |

### first20

- Best mean Dice: `DirectCP 8-4-2 alpha1000` with Dice `0.8077`.
- New mean Dice/HD95/symmetric ASD: `0.8077` / `2.2589` / `0.4531`.
- New foreground native BE/logJ std/fold%: `0.1150` / `0.9038` / `4.4071`.

| Method | n | alpha | lr | dice mean | hd95 mean | asd warped to fixed mean | asd fixed to warped mean | asd symmetric mean | be fd forward fixed fg voxel | be native analytic fixed fg voxel | diffusion l2 fg | diffusion fd forward fixed fg voxel | diffusion native analytic fixed fg voxel | log jdet std fg | jdet le0 pct fg | ndv frac fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IDIR best Dice | 20 | 1 | 1.000e-04 | 0.7751 | 2.3639 | 0.5444 | 0.4700 | 0.5261 | 0.0307 | 0.0312 | 0.1068 | 0.1068 | 0.1071 | 0.7437 | 2.2690 | 0.0058 |
| SINR cps4 alpha1000 | 20 | 1e+03 | 1.000e-04 | 0.7962 | 2.3196 | 0.5124 | 0.4041 | 0.4778 | 0.0908 | 0.0956 | 0.1902 | 0.1902 | 0.1912 | 0.9517 | 5.6781 | 0.0197 |
| SINR cps2 alpha1000 | 20 | 1e+03 | 1.000e-04 | 0.8004 | 2.2664 | 0.4899 | 0.3919 | 0.4580 | 0.0962 | 0.1019 | 0.1850 | 0.1850 | 0.1861 | 0.9375 | 5.4076 | 0.0193 |
| DirectCP cps4 alpha1000 | 20 | 1e+03 | 0.00100 | 0.7984 | 2.3223 | 0.5118 | 0.3884 | 0.4693 | 0.1129 | 0.1199 | 0.1895 | 0.1895 | 0.1909 | 0.9357 | 5.8050 | 0.0231 |
| DirectCP cps2 alpha1000 | 20 | 1e+03 | 0.00100 | 0.8033 | 2.3716 | 0.5223 | 0.3743 | 0.4713 | 0.1295 | 0.1442 | 0.1825 | 0.1825 | 0.1840 | 0.9324 | 5.8441 | 0.0261 |
| DirectCP 8-4-2 alpha1000 | 20 | 1e+03 | 0.01000 | 0.8077 | 2.2589 | 0.4900 | 0.3777 | 0.4531 | 0.1031 | 0.1150 | 0.1668 | 0.1668 | 0.1681 | 0.9038 | 4.4071 | 0.0152 |

## Runtime

Runtime table for `first20`.

| Method | n | alpha | lr | training elapsed sec | total elapsed sec | training peak mem gb |
| --- | --- | --- | --- | --- | --- | --- |
| IDIR best Dice | 20 | 1 | 1.000e-04 | 122.8008 | 130.7589 | 4.1227 |
| SINR cps4 alpha1000 | 20 | 1e+03 | 1.000e-04 | 138.8179 | 146.0403 | 2.4996 |
| SINR cps2 alpha1000 | 20 | 1e+03 | 1.000e-04 | 304.8342 | 312.0033 | 7.6328 |
| DirectCP cps4 alpha1000 | 20 | 1e+03 | 0.00100 | 111.7774 | 118.9626 | 1.7507 |
| DirectCP cps2 alpha1000 | 20 | 1e+03 | 0.00100 | 117.1771 | 124.0482 | 2.0855 |
| DirectCP 8-4-2 alpha1000 | 20 | 1e+03 | 0.01000 | 194.1250 | 232.8295 | 3.9319 |

## Files

- `comparison_population_first10_comprehensive.csv/json` and `comparison_population_first10_main.csv`: first10 comparison rows.
- `comparison_population_first20_comprehensive.csv/json` and `comparison_population_first20_main.csv`: first20 comparison rows, when ranks `0-19` are available.
- `comparison_population_comprehensive.csv/json`: compatibility copy of the largest completed/requested scope.
- `summary_runs_new.csv/json`: new per-case summaries for the requested ranks.
- `stage_history_new.csv`, `stage_summaries_new.csv`, and `prolongation_metrics_new.csv`: multistage diagnostics.
