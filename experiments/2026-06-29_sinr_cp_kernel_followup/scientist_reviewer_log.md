# Scientist / Reviewer Exploration Log

Goal: produce a useful paper table, figure, or section from an exploratory SINR/DirectCP follow-up.

## 2026-06-29 09:04:31 +0000 - Reviewer

Treat the failed promising gate as information, not as a stop. Proceed with exploration, but do not claim a method-only LungCT leaderboard. Ask whether low SINR-vs-DirectCP similarity reflects a real mechanism difference, a capture-range limitation, or a schedule/objective artifact.

## 2026-06-29 09:04:31 +0000 - Scientist

Plan: rerun the sampled dense-update proxy on the completed full outputs; combine it with prior case01/case08 field-fit and LR sweeps; then launch schedule ablations that remove or weaken the image pyramid while preserving the DirectCP control-stage ladder.

## 2026-06-29 09:04:31 +0000 - Reviewer

Approved launch of `10` schedule-ablation jobs. Require aggregation after completion.

## 2026-06-29 09:05:00 +0000 - Reviewer

Treat the failed promising gate as information, not as a stop. Proceed with exploration, but do not claim a method-only LungCT leaderboard. Ask whether low SINR-vs-DirectCP similarity reflects a real mechanism difference, a capture-range limitation, or a schedule/objective artifact.

## 2026-06-29 09:05:00 +0000 - Scientist

Plan: rerun the sampled dense-update proxy on the completed full outputs; combine it with prior case01/case08 field-fit and LR sweeps; then launch schedule ablations that remove or weaken the image pyramid while preserving the DirectCP control-stage ladder.

## 2026-06-29 09:05:00 +0000 - Reviewer

Approved launch of `4` schedule-ablation jobs. Require aggregation after completion.

## 2026-06-29 09:11:11 +0000 - Scientist

Schedule launch result: `{'status': 'completed', 'completed': [{'job': 'explore_case01_ptv_nopyr_ds1-1-1-1_e1-1-1-1_lr0p001_w0p11', 'group': 'ptv_control_staging_without_image_pyramid', 'variant': 'ptv_nopyr_ds1-1-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 250.01675724983215}, {'job': 'explore_case01_fdbe_nopyr_ds1-1-1-1_e1-1-1-1_lr0p001_a10000', 'group': 'fdbe_control_staging_without_image_pyramid', 'variant': 'fdbe_nopyr_ds1-1-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 248.0139617919922}, {'job': 'explore_case08_fdbe_nopyr_ds1-1-1-1_e1-1-1-1_lr0p001_a10000', 'group': 'fdbe_control_staging_without_image_pyramid', 'variant': 'fdbe_nopyr_ds1-1-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 334.62108302116394}, {'job': 'explore_case08_ptv_nopyr_ds1-1-1-1_e1-1-1-1_lr0p001_w0p11', 'group': 'ptv_control_staging_without_image_pyramid', 'variant': 'ptv_nopyr_ds1-1-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 367.14635848999023}], 'failures': [], 'elapsed_sec': 371.16384053230286}`.

## 2026-06-29 09:11:12 +0000 - Scientist

Aggregation refreshed: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 4, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:12:46 +0000 - Reviewer

Treat the failed promising gate as information, not as a stop. Proceed with exploration, but do not claim a method-only LungCT leaderboard. Ask whether low SINR-vs-DirectCP similarity reflects a real mechanism difference, a capture-range limitation, or a schedule/objective artifact.

## 2026-06-29 09:12:46 +0000 - Scientist

Plan: rerun the sampled dense-update proxy on the completed full outputs; combine it with prior case01/case08 field-fit and LR sweeps; then launch schedule ablations that remove or weaken the image pyramid while preserving the DirectCP control-stage ladder.

## 2026-06-29 09:12:48 +0000 - Scientist

Proxy follow-up completed: `0`.

## 2026-06-29 09:12:48 +0000 - Reviewer

Approved launch of `10` schedule-ablation jobs. Require aggregation after completion.

## 2026-06-29 09:15:35 +0000 - Reviewer

Sidecar monitor started. It must not stop or relaunch jobs; it should refresh partial artifacts only when completed schedule rows exist.

## 2026-06-29 09:15:35 +0000 - Scientist

Monitor status: `running`, completed `0`, failures `0`, running `4`, pending `6`.

## 2026-06-29 09:16:57 +0000 - Reviewer

Smoke plus early live logs confirm that full no-pyramid all10 would not be the right next unit. Keep the current run focused on cases `01` and `08`; use the partial-pyramid rows to ask whether capture range, not representation capacity, explains the case08 rescue.

## 2026-06-29 09:16:57 +0000 - Scientist

Proceeding with the active 10-job targeted schedule. The paper artifact should compare: low SINR/DirectCP update similarity, case08 field-fit capacity, standard multistage success, no-pyramid failure, and partial-pyramid recovery if it appears.

## 2026-06-29 09:19:04 +0000 - Scientist

Manual partial aggregation after first long-run completions: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:20:12 +0000 - Scientist

Paper-ready mechanism table refreshed after aggregator patch: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:20:29 +0000 - Scientist

Partial aggregation refreshed after both case08 no-pyramid completions: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:24:03 +0000 - Scientist

Partial aggregation refreshed at 6/10 schedule completions: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:24:22 +0000 - Scientist

Partial aggregation refreshed at 8/10 schedule completions: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:29:13 +0000 - Scientist

Partial aggregation refreshed after case08 pTV partial-pyramid completion: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:29:13 +0000 - Scientist

Schedule launch result: `{'status': 'completed', 'completed': [{'job': 'explore_case01_ptv_nopyr_ds1-1-1-1_e25-25-25-25_lr0p001_w0p11', 'group': 'ptv_control_staging_without_image_pyramid', 'variant': 'ptv_nopyr_ds1-1-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 279.81216835975647}, {'job': 'explore_case01_fdbe_nopyr_ds1-1-1-1_e25-25-25-25_lr0p001_a10000', 'group': 'fdbe_control_staging_without_image_pyramid', 'variant': 'fdbe_nopyr_ds1-1-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 277.80930042266846}, {'job': 'explore_case08_fdbe_nopyr_ds1-1-1-1_e25-25-25-25_lr0p001_a10000', 'group': 'fdbe_control_staging_without_image_pyramid', 'variant': 'fdbe_nopyr_ds1-1-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 367.91821002960205}, {'job': 'explore_case08_ptv_nopyr_ds1-1-1-1_e25-25-25-25_lr0p001_w0p11', 'group': 'ptv_control_staging_without_image_pyramid', 'variant': 'ptv_nopyr_ds1-1-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 401.9582886695862}, {'job': 'explore_case01_ptv_nopyr_ds1-1-1-1_lr0p01_e25-25-25-25_lr0p01_w0p11', 'group': 'ptv_no_pyramid_best_lr_probe', 'variant': 'ptv_nopyr_ds1-1-1-1_lr0p01', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 278.3256390094757}, {'job': 'explore_case01_ptv_partial_ds4-2-1-1_e25-25-25-25_lr0p001_w0p11', 'group': 'ptv_partial_image_pyramid_probe', 'variant': 'ptv_partial_ds4-2-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 246.28284406661987}, {'job': 'explore_case08_ptv_nopyr_ds1-1-1-1_lr0p01_e25-25-25-25_lr0p01_w0p11', 'group': 'ptv_no_pyramid_best_lr_probe', 'variant': 'ptv_nopyr_ds1-1-1-1_lr0p01', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 370.432626247406}, {'job': 'explore_case01_fdbe_partial_ds4-2-1-1_e25-25-25-25_lr0p001_a10000', 'group': 'fdbe_partial_image_pyramid_probe', 'variant': 'fdbe_partial_ds4-2-1-1', 'case_id': 1, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 246.28380918502808}, {'job': 'explore_case08_ptv_partial_ds4-2-1-1_e25-25-25-25_lr0p001_w0p11', 'group': 'ptv_partial_image_pyramid_probe', 'variant': 'ptv_partial_ds4-2-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 336.37771439552307}, {'job': 'explore_case08_fdbe_partial_ds4-2-1-1_e25-25-25-25_lr0p001_a10000', 'group': 'fdbe_partial_image_pyramid_probe', 'variant': 'fdbe_partial_ds4-2-1-1', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 304.3382227420807}], 'failures': [], 'elapsed_sec': 985.1147041320801}`.

## 2026-06-29 09:29:14 +0000 - Scientist

Aggregation refreshed: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:29:14 +0000 - Reviewer

Long-run supervision requested; waiting 1800 sec before the next aggregation pass. If new rows are complete, the Scientist should refresh tables and paper section.

## 2026-06-29 09:31:07 +0000 - Scientist

Final first-phase aggregation after 10/10 schedule completions and patched artifact writer: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:33:05 +0000 - Reviewer

Second-phase instruction: because no-pyramid and partial `4,2,1,1` did not rescue case08, launch case08 standard `8,4,2,1` probes to separate image-pyramid capture range, LR, objective, and modest schedule length.

## 2026-06-29 09:33:05 +0000 - Scientist

Standard-pyramid probe launch result: `{'status': 'dry_run', 'job_count': 4}`.

## 2026-06-29 09:33:05 +0000 - Scientist

Standard-pyramid probe aggregation refreshed: `{'status': 'completed', 'rows': 4, 'completed_rows': 0, 'best_completed': None, 'rows_csv': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables/standard_pyramid_probe_rows.csv', 'report': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/standard_pyramid_probe_report.md'}`.

## 2026-06-29 09:33:13 +0000 - Reviewer

Second-phase instruction: because no-pyramid and partial `4,2,1,1` did not rescue case08, launch case08 standard `8,4,2,1` probes to separate image-pyramid capture range, LR, objective, and modest schedule length.

## 2026-06-29 09:34:45 +0000 - Scientist

Actual second-phase standard-pyramid probe is running detached: 4 case08 jobs active on GPUs `0,1,2,3`, status file `standard_pyramid_probe/phase_status.json` reports `running=4`, `failures=0`.

## 2026-06-29 09:38:52 +0000 - Scientist

Standard-pyramid probe launch result: `{'status': 'completed', 'completed': [{'job': 'stdprobe_case08_ptv_standard_short_ds8-4-2-1_e25-25-25-25_lr0p001_w0p11', 'group': 'standard_pyramid_short_ptv', 'variant': 'ptv_standard_short', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 308.3339419364929}, {'job': 'stdprobe_case08_fdbe_standard_short_ds8-4-2-1_e25-25-25-25_lr0p001_a10000', 'group': 'standard_pyramid_short_fdbe', 'variant': 'fdbe_standard_short', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_global_ncc_fd_be', 'returncode': 0, 'elapsed_sec': 306.3310842514038}, {'job': 'stdprobe_case08_ptv_standard_e50_ds8-4-2-1_e50-50-50-50_lr0p001_w0p11', 'group': 'standard_pyramid_longer_ptv', 'variant': 'ptv_standard_e50', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 302.3252408504486}, {'job': 'stdprobe_case08_ptv_standard_short_lr0p01_ds8-4-2-1_e25-25-25-25_lr0p01_w0p11', 'group': 'standard_pyramid_lr_probe_ptv', 'variant': 'ptv_standard_short_lr0p01', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 334.36171436309814}], 'failures': [], 'elapsed_sec': 338.822847366333}`.

## 2026-06-29 09:38:52 +0000 - Scientist

Standard-pyramid probe aggregation refreshed: `{'status': 'completed', 'rows': 4, 'completed_rows': 4, 'best_completed': {'group': 'standard_pyramid_lr_probe_ptv', 'variant': 'ptv_standard_short_lr0p01', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'run_name': 'stdprobe_case08_ptv_standard_short_lr0p01_ds8-4-2-1_e25-25-25-25_lr0p01_w0p11', 'status': 'completed', 'stage_downsample': '8,4,2,1', 'stage_epochs': '25,25,25,25', 'stage_lr': '0.01,0.01,0.01,0.01', 'loss': 'pTV-style LCC + isoTV', 'alpha': None, 'ptv_isotv_weight': 0.11, 'reg1mm_snap_tre_mean_mm': 2.2704644993422045, 'reg1mm_subvoxel_tre_mean_mm': 2.151804789336156, 'elapsed_sec': 265.9466416835785, 'peak_mem_gb': 6.493811130523682, 'summary_path': '/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1/standard_pyramid_probe/direct_cp_32_16_8_4_ptv_lcc_isotv/stdprobe_case08_ptv_standard_short_lr0p01_ds8-4-2-1_e25-25-25-25_lr0p01_w0p11/summary.json'}, 'rows_csv': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables/standard_pyramid_probe_rows.csv', 'report': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/standard_pyramid_probe_report.md'}`.

## 2026-06-29 09:41:29 +0000 - Scientist

Final combined mechanism aggregation with standard-pyramid rows: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'standard_probe_rows': 4, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:42:46 +0000 - Scientist

Final combined aggregation regenerated with metric-matched case08 plot: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'standard_probe_rows': 4, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:43:26 +0000 - Scientist

Second-phase result integrated: standard `8,4,2,1` case08 probe completed `4/4` with `0` failures. Best row is pTV LCC + isoTV, `25,25,25,25`, LR `0.01`, reg1mm snap TRE `2.270` mm. This is a large recovery over no-pyramid LR `0.01` (`6.478` mm) and partial `4,2,1,1` (`11.937` mm).

## 2026-06-29 09:43:26 +0000 - Reviewer

Interpretation for paper artifact: frame the result as mechanism evidence on selected diagnostic cases. The data support representation capacity being sufficient, while hard-case image registration depends on capture range plus optimizer scale/runtime. Do not present this as a population-final LungCT method table, and keep metric-convention caveats explicit.

## 2026-06-29 09:44:46 +0000 - Reviewer

Refinement instruction: launch case08 LR/runtime controls after standard-pyramid recovery.

## 2026-06-29 09:44:46 +0000 - Scientist

Capture-refinement probe launch result: `{'status': 'dry_run', 'job_count': 4}`.

## 2026-06-29 09:44:46 +0000 - Scientist

Capture-refinement probe aggregation refreshed: `{'status': 'completed', 'rows': 4, 'completed_rows': 0, 'best_completed': None, 'rows_csv': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables/capture_refinement_probe_rows.csv', 'report': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/capture_refinement_probe_report.md'}`.

## 2026-06-29 09:44:56 +0000 - Reviewer

Refinement instruction: launch case08 LR/runtime controls after standard-pyramid recovery.

## 2026-06-29 09:46:33 +0000 - Scientist

Capture-refinement probe is running detached on GPUs `0,1,2,3`: standard `8,4,2,1` LR `0.01` with e100/e200 schedules, plus partial `4,2,1,1` LR `0.01` e100 and no-pyramid LR `0.01` e100 controls. Initial status: `running=4`, `failures=0`.

## 2026-06-29 09:46:33 +0000 - Reviewer

Use the refinement rows to decide whether the remaining gap after standard-pyramid recovery is mostly runtime, pyramid extent, or both. Preserve the metric caveat and keep this as diagnostic mechanism evidence.

## 2026-06-29 09:52:06 +0000 - Scientist

Capture-refinement probe launch result: `{'status': 'completed', 'completed': [{'job': 'stdprobe_case08_ptv_standard_lr0p01_e100_ds8-4-2-1_e100-100-100-100_lr0p01_w0p11', 'group': 'standard_pyramid_runtime_probe_ptv', 'variant': 'ptv_standard_lr0p01_e100', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 278.3171315193176}, {'job': 'stdprobe_case08_ptv_standard_lr0p01_e200_ds8-4-2-1_e200-200-200-200_lr0p01_w0p11', 'group': 'standard_pyramid_runtime_probe_ptv', 'variant': 'ptv_standard_lr0p01_e200', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 306.3483474254608}, {'job': 'stdprobe_case08_ptv_partial_lr0p01_e100_ds4-2-1-1_e100-100-100-100_lr0p01_w0p11', 'group': 'partial_pyramid_lr_runtime_probe_ptv', 'variant': 'ptv_partial_lr0p01_e100', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 364.9275221824646}, {'job': 'stdprobe_case08_ptv_nopyr_lr0p01_e100_ds1-1-1-1_e100-100-100-100_lr0p01_w0p11', 'group': 'no_pyramid_lr_runtime_probe_ptv', 'variant': 'ptv_nopyr_lr0p01_e100', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'returncode': 0, 'elapsed_sec': 423.47641372680664}], 'failures': [], 'elapsed_sec': 429.9859366416931}`.

## 2026-06-29 09:52:06 +0000 - Scientist

Capture-refinement probe aggregation refreshed: `{'status': 'completed', 'rows': 4, 'completed_rows': 4, 'best_completed': {'group': 'standard_pyramid_runtime_probe_ptv', 'variant': 'ptv_standard_lr0p01_e100', 'case_id': 8, 'method': 'direct_cp_32_16_8_4_ptv_lcc_isotv', 'run_name': 'stdprobe_case08_ptv_standard_lr0p01_e100_ds8-4-2-1_e100-100-100-100_lr0p01_w0p11', 'status': 'completed', 'stage_downsample': '8,4,2,1', 'stage_epochs': '100,100,100,100', 'stage_lr': '0.01,0.01,0.01,0.01', 'loss': 'pTV-style LCC + isoTV', 'alpha': None, 'ptv_isotv_weight': 0.11, 'reg1mm_snap_tre_mean_mm': 1.2134897659389092, 'reg1mm_subvoxel_tre_mean_mm': 1.083705282897508, 'elapsed_sec': 232.8840606212616, 'peak_mem_gb': 6.493811130523682, 'summary_path': '/database/inrorspline/runs/sinr_cp_mechanism_explore_20260629_v1/capture_refinement_probe/direct_cp_32_16_8_4_ptv_lcc_isotv/stdprobe_case08_ptv_standard_lr0p01_e100_ds8-4-2-1_e100-100-100-100_lr0p01_w0p11/summary.json'}, 'rows_csv': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables/capture_refinement_probe_rows.csv', 'report': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/capture_refinement_probe_report.md'}`.

## 2026-06-29 09:53:15 +0000 - Scientist

Final combined aggregation regenerated with capture-refinement rows: `{'status': 'completed', 'prior_rows': 16, 'schedule_rows': 10, 'standard_probe_rows': 4, 'refinement_probe_rows': 4, 'update_similarity_rows': 3, 'proxy_rows': 9, 'tables_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/tables', 'plots_dir': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/plots', 'paper_section': '/homebase/code_sync/inrorspline/experiments/2026-06-29_sinr_cp_kernel_followup/exploration/paper_section_draft.md'}`.

## 2026-06-29 09:54:33 +0000 - Scientist

Capture-refinement probe completed `4/4` with `0` failures. Case08 LR `0.01` e100/e200 controls converge near `1.21` mm reg1mm snap TRE: standard e100 `1.213`, standard e200 `1.217`, partial e100 `1.219`, no-pyramid e100 `1.218`.

## 2026-06-29 09:54:33 +0000 - Reviewer

Final interpretation: the result is stronger and more nuanced than the short-budget ablation. The pyramid helps early capture under small budgets, but the final diagnostic recovery is driven mainly by optimizer scale/runtime once LR is set to `0.01`. The paper section should emphasize optimization path and budget rather than a fixed pyramid requirement.
