# OASIS Seed-3 First100 Extension Progress Log

- Updated: `2026-07-06 14:10:03 +0000`
- Progress log: `/homebase/code_sync/inrorspline/experiments/2026-07-04_oasis_seed3_first100_extension/progress_log.md`
- Tracker figure: `/homebase/code_sync/inrorspline/experiments/2026-07-04_oasis_seed3_first100_extension/tracker/oasis_first100_progress_tracker.png`
- Final artifact root: `/homebase/code_sync/inrorspline/experiments/2026-07-04_oasis_seed3_first100_extension/first100_artifacts`
- Baseline extension root: `/database/inrorspline/runs/oasis_seed3_ranks020_099_saved_baselines_v1`
- Single-stage extension root: `/database/inrorspline/runs/oasis_seed3_ranks020_099_alpha_sweep_be_2500e_5models_v1`
- MR-D-BSCP extension root: `/database/inrorspline/runs/oasis_seed3_ranks020_099_directcp842_lr1e3_alpha_sweep_v1`
- Metric convention: absent fixed, moving, or warped label voxels produce `NaN`; population means use finite-only `nanmean`.

## Overall

- First100-complete method/hyperparameter rows: `41/41`
- New rank20-99 jobs done: `3280/3280`
- Running jobs: `0`
- Failed jobs: `0`
- Pending new jobs: `0`

## Run Matrix Status

| Status | Stage | Method | Alpha | LR | Cases done | New done | Running | Failed | Pending |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| done | reference eval | `bs1_vxm_v0_2` | baseline |  | 100/100 | 80/80 | 0 | 0 | 0 |
| done | reference eval | `bs2_tm_v0` | baseline |  | 100/100 | 80/80 | 0 | 0 | 0 |
| done | reference eval | `bs3_vfa_v0` | baseline |  | 100/100 | 80/80 | 0 | 0 | 0 |
| done | reference eval | `bs4_sitreg_v0` | baseline |  | 100/100 | 80/80 | 0 | 0 | 0 |
| done | reference eval | `greedy` | baseline |  | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 0 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 10 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 100 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 1000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 10000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | MR-D-BSCP 8-4-2 | `direct_cp_8_4_2_be_bspline_analytic_multistage` | 100000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 0 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 10 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 100 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 1000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 10000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps2) | `direct_cp_cps2_be_bspline_analytic` | 100000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 0 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 10 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 100 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 1000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 10000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps2) | `sinr_cps2_be_bspline_analytic` | 100000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 0 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 0.01 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 0.1 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 1 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 10 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-Dense | `idir_original_dense_siren_be_autograd` | 100 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 0 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 10 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 100 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 1000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 10000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | D-BSCP (cps4) | `direct_cp_cps4_be_bspline_analytic` | 100000 | 0.001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 0 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 10 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 100 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 1000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 10000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |
| done | INR-BSCP (cps4) | `sinr_cps4_be_bspline_analytic` | 100000 | 0.0001 | 100/100 | 80/80 | 0 | 0 | 0 |

## Completed Extension Rows

- Saved baselines rank20-99 summaries: `400/400`
- MR-D-BSCP 8-4-2 rank20-99 summaries: `480/480`
- Single-stage rank20-99 summaries: `2400/2400`

## Failure Logs

- none
