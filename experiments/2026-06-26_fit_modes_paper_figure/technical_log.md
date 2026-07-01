# Fit-Modes Paper Figure Technical Log

## Status

- Planned and implemented on 2026-06-26.
- Docker target: `py310sitreg_v1_test1`, image
  `hengjieliu96/dl_dir:py310sitreg_v1`.
- Requested full launch GPUs: CUDA `0,1,2` in parallel.

## Known Limitations Before Launch

- These reruns add diagnostic image-similarity evaluation curves for IDIR only.
  IDIR training remains its original sampled NCC plus bending-energy objective.
- The eval curves are dataset-specific, not one universal similarity metric:
  OASIS uses full-image global NCC, while LungCT uses pTV LCC data loss on the
  border-masked pTVReg crop to match the SINR/DirectCP LungCT histories.
- The three reruns are single selected operating points: OASIS IDIR alpha `1`
  and LungCT IDIR `alpha_bending=10`, all at lr `1e-4`.
- Existing accepted OASIS and LungCT fit-mode outputs are read-only inputs for
  the later notebook and must not be overwritten.

## Acceptance Checks Planned

- Modified OASIS and LungCT runners compile inside Docker.
- A 20-epoch smoke run writes `image_similarity_eval.csv` for all three IDIR
  cases with epochs `1,10,20`.
- Full run writes `image_similarity_eval.csv` for all three IDIR cases with
  epochs `1,10,20,...,2500`.
- Each full run completes and writes the usual summary/dense-field artifacts.

## Launch Log

- 2026-06-26: compiled modified runners inside Docker:
  `scripts/experiments/oasis_alpha_sweep/run_one.py` and
  `scripts/experiments/lungct_no_sweep_repro/run_one.py`.
- 2026-06-26: 20-epoch smoke run passed in
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_smoke_v1`.
  All three `image_similarity_eval.csv` files contained epochs `1,10,20`.
- 2026-06-26: an initial full launch used buffered Python output and was moved
  aside as
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1_aborted_buffered_20260626T184103Z`
  so milestone monitoring would be real-time.
- 2026-06-26: full unbuffered launch completed in
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1`.
  Launcher returned `full_complete`.
- Full-run CSV validation passed: all three eval CSVs exist, each has `251`
  rows, and each covers epoch `1` plus every `10` epochs through `2500`
  without missing or extra epochs.

## 2026-07-01 OASIS DirectCP 8-4-2 Field-Fit Panel B Add-On

- Purpose: add OASIS DirectCP 8-4-2 multistage field-fit trajectories to the
  standardized Panel B rRMSE fitting figure for both SITReg-target and
  Greedy-target OASIS rank9 panels.
- Method: `direct_cp_8_4_2_be_bspline_analytic_multistage`, stored under
  field-fit key `direct_cp_8_4_2_field_mse`.
- Case: seed-3 shuffled OASIS rank `9`, original sorted index `88`,
  fixed/moving `(0173,0136)`.
- Schedule: cps `8,4,2`; stage downsample `4,2,1`; stage epochs
  `400,800,1300`; total `2500` epochs; lr `1e-3`; alpha `1000` recorded for
  Panel A consistency but unused by the field-MSE objective.
- Optimized loss: stage-domain normalized-grid target-field MSE over the fixed
  raw image-positive brain mask; no image loss and no regularization.
- Diagnostic loss: finest-domain masked normalized-grid target-field MSE at
  epoch `1`, every `10` global epochs, and final epoch. The standardized Panel
  B curve uses `rRMSE = 100 * sqrt(mse / (target_power + 1e-12))`.
- Output root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_v1`.
- Smoke root:
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_smoke_v1`.
- Planned artifact update:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_rrmse_field_fit_curves.png`.
- Result: Docker `py_compile` passed; both three-epoch smoke jobs passed; both
  full 2500-epoch jobs completed with `DONE` markers under
  `/database/inrorspline/runs/fit_modes_oasis_directcp842_field_fit_v1`.
- Diagnostic validation: both full histories have 2500 rows and 251 finite
  finest-domain MSE diagnostics at epoch `1` plus every `10` epochs through
  `2500`.
- Final Panel B rRMSE values:
  - SITReg target: start `97.33%`, epoch 10 `76.30%`, final `4.50%`.
  - Greedy target: start `97.58%`, epoch 10 `78.18%`, final `1.23%`.
- Standardized Panel B artifact regenerated with the OASIS DirectCP 8-4-2
  curves included:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_rrmse_field_fit_curves.png`.

## Full Run Outputs

- OASIS:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/oasis_rank009/idir_original_dense_siren_be_autograd/oasis_rank009_idir_a1_lr1e-4_global_ncc_eval/image_similarity_eval.csv`
  - Eval metric: full-image global NCC.
  - Epoch 10 loss: `-0.9032629728`.
  - Epoch 100 loss: `-0.9559287429`.
  - Epoch 200 loss: `-0.9609792829`.
  - Epoch 2500 loss: `-0.9750214815`; NCC value `0.9750214815`.
  - Final Dice mean from summary: `0.8110367511`.
- LungCT case01:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case01/idir_be_default_lungmask/lungct_case01_idir_lr1e-4_ptv_lcc_eval/image_similarity_eval.csv`
  - Eval metric: border-masked pTV LCC data loss.
  - Epoch 10 loss: `4551105.5`.
  - Epoch 100 loss: `2945024.0`.
  - Epoch 200 loss: `2872192.5`.
  - Epoch 2500 loss: `2817243.5`.
  - Summary elapsed: `272.72` seconds; peak memory: `4.26` GB.
  - Final subvoxel TRE mean: `0.9425751087` mm.
- LungCT case08:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case08/idir_be_default_lungmask/lungct_case08_idir_lr1e-4_ptv_lcc_eval/image_similarity_eval.csv`
  - Eval metric: border-masked pTV LCC data loss.
  - Epoch 10 loss: `8923448.0`.
  - Epoch 100 loss: `6462548.0`.
  - Epoch 200 loss: `5801685.0`.
  - Epoch 2500 loss: `5032755.5`.
  - Summary elapsed: `316.27` seconds; peak memory: `4.53` GB.
  - Final subvoxel TRE mean: `1.2343288412` mm.

## Notebook Figure Outputs

- 2026-06-26: added and executed
  `fit_modes_method_comparison.ipynb` as the primary editable review surface.
- The notebook compares selected IDIR, SINR cps4, DirectCP cps4, and available
  LungCT DirectCP 4-stage curves across OASIS rank-9, LungCT case01, and
  LungCT case08.
- Registration panels use matched image similarity:
  - OASIS: global NCC loss.
  - LungCT: pTV LCC data loss.
  - IDIR registration is sparse by design: epoch `1` plus every `10` epochs.
- Field-fit panels use objective masked normalized-grid MSE against SITReg
  (OASIS) or pTVReg (LungCT) target fields.
- Figure and support CSVs:
  - `fit_modes_method_comparison_curves.pdf`
  - `fit_modes_method_comparison_curves.png`
  - `fit_modes_method_comparison_selected_runs.csv`
  - `fit_modes_method_comparison_curve_summary.csv`
- Verification: notebook executed in Docker with populated outputs and no
  notebook errors; helper scripts pass `py_compile`; `git diff --check` passes.
- 2026-06-26 correction: updated the notebook figure so LungCT DirectCP
  4-stage registration uses the existing posthoc `finest_data_loss`
  diagnostic curves instead of stage-domain training `data_loss`. Updated
  LungCT DirectCP 4-stage field-fit plotting to show only the final
  `downsample=1` finest-domain segment because existing coarse-stage MSE
  histories are stage-domain and do not contain full-domain per-epoch MSE.
- 2026-06-26 planned add-on: rerun LungCT case01/case08 DirectCP 4-stage
  pTVReg field fitting with a new posthoc finest-domain MSE diagnostic. Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
  This will replace the final-stage-only workaround in the notebook figure.
- 2026-06-26 completion: reran LungCT case01/case08 DirectCP 4-stage pTVReg
  field fitting under the same training schedule and lr `1e-2`. To keep the
  dense diagnostic affordable, coarse stages log posthoc finest-domain MSE
  every 50 epochs and the final 1mm stage logs it every epoch, giving `935`
  finite `finest_data_loss` points per case over `2500` history rows. Final
  values are case01 `2.846068e-08` and case08 `9.682780e-08`. The executed
  notebook now uses these curves instead of the prior final-stage-only
  workaround for LungCT DirectCP 4-stage field fitting.

## rRMSE LR Trajectory Add-On

- 2026-07-01 planned add-on: build a field-fit-only rRMSE trajectory figure
  comparing requested LRs for IDIR (`1e-3`, `1e-4`), SINR cps4 (`1e-3`,
  `1e-4`), DirectCP cps4 (`1e-2`, `1e-3`, `1e-4`), and DirectCP 4-stage
  (`1e-2`, `1e-3`, `1e-4`).
- Existing OASIS single-stage, OASIS Greedy-field, LungCT single-stage, and
  LungCT DirectCP 4-stage `lr=1e-2` finest-domain diagnostic histories are
  reused. New reruns are limited to LungCT DirectCP 4-stage pTVReg-field
  fitting at case01/case08 x lr `1e-3`/`1e-4`.
- Training protocol for these reruns remains unchanged: DirectCP 4-stage
  pTVReg-field MSE, schedule `32,16,8,4`, downsample `8,4,2,1`, stage epochs
  `400,400,800,900`, 2500 epochs, seed `1`. The new curve values use posthoc
  finest-domain masked normalized-grid MSE diagnostics only.
- Planned output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1`.
- Planned repo artifacts:
  `fit_modes_rrmse_lr_trajectory_curves.{png,pdf}`,
  `fit_modes_rrmse_lr_trajectory_curve_summary.csv`, and
  `fit_modes_rrmse_lr_trajectory_selected_runs.csv`.
- Acceptance checks: two-epoch smoke with finite finest-domain columns; four
  full histories with `2500` rows and finite finest-domain diagnostic points;
  rRMSE figure has shared log y-axis capped at `100%`; notebook executes in
  Docker; helper scripts pass `py_compile`; `git diff --check` passes.
- 2026-07-01 smoke completed in
  `/database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_smoke_v1`.
  The two-row `stage_history.csv` had finite `finest_data_loss` and
  `finest_field_mse_objective_normalized_grid` values at epochs `1` and `2`.
- 2026-07-01 full diagnostic reruns completed for LungCT case01/case08 at
  lr `1e-3` and `1e-4`. Each `stage_history.csv` has `2500` rows and `935`
  finite finest-domain diagnostic points. Final finest-domain history values:
  case01 lr `1e-3` `2.894892e-08`, case01 lr `1e-4` `3.456253e-08`,
  case08 lr `1e-3` `9.883180e-08`, case08 lr `1e-4` `3.594418e-05`.
  These are within `1%` of the corresponding original sweep final summaries;
  the small residual mismatch reflects pre-step history rows versus post-step
  summary metrics.
- 2026-07-01 notebook execution wrote
  `fit_modes_rrmse_lr_trajectory_curves.{png,pdf}`,
  `fit_modes_rrmse_lr_trajectory_curve_summary.csv`, and
  `fit_modes_rrmse_lr_trajectory_selected_runs.csv`. The selected-runs CSV has
  `40` requested rows: `34` available curves and `6` explicit OASIS
  DirectCP 4-stage not-run entries. The figure uses a shared log rRMSE percent
  y-axis capped at `100%`.
- Final verification passed: Docker `py_compile` for the figure generator,
  notebook generator, and LungCT field-fit runner; generated CSV sanity checks;
  visual inspection of the rendered PNG; and `git diff --check`.

## Panel A Global-NCC Refresh

- 2026-07-01 planned add-on: rebuild Panel A as a global-NCC loss trajectory
  comparison across OASIS rank9, LungCT case01, and LungCT case08. The old
  figure mixed OASIS global NCC and LungCT pTV LCC; this refresh makes the
  y-axis comparable across all three cases.
- Existing OASIS curves reused:
  - IDIR alpha `1`, lr `1e-4`, 2500 epochs, with the completed full-image
    global-NCC diagnostic every 10 epochs.
  - SINR cps4 alpha `1000`, lr `1e-4`, 2500 epochs, optimized full-image
    global NCC + analytic B-spline BE.
  - DirectCP cps4 alpha `1000`, lr `1e-3`, 2500 epochs, optimized full-image
    global NCC + analytic B-spline BE.
- New OASIS curve to run: DirectCP 8-4-2, alpha `1000`, lr `1e-3`, cps
  `8,4,2`, downsample `4,2,1`, stage epochs `400,800,1300`, total 2500
  epochs. This matches the updated canonical OASIS multistage setting requested
  for paper Panel A.
- New LungCT curves to run for case01 and case08:
  - IDIR `alpha_bending=3`, lr `1e-4`, 2500 epochs. Training stays native IDIR
    sampled NCC + BE; plotted diagnostic is full-crop global NCC every 10
    epochs.
  - SINR cps4 global-NCC+FD-BE, alpha `10000`, lr `1e-4`, 2500 epochs.
  - DirectCP cps4 global-NCC+FD-BE, alpha `10000`, lr `1e-3`, 2500 epochs.
  - DirectCP 32-16-8-4 global-NCC+FD-BE, alpha `10000`, lr `1e-3`, cps
    `32,16,8,4`, downsample `8,4,2,1`, stage epochs `200,400,800,1000`,
    total 2400 epochs to match the existing Pareto protocol.
- Planned output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1`.
- Planned standardized artifact:
  `experiments/paper_artifacts/fitting_figure/v001/fit_modes_standardized_global_ncc_panelA_curves.png`.
- Figure rule: share the global NCC loss y-axis across all three cases and
  force y tick labels plus y-axis label visible on every subplot.
- 2026-07-01 completion: Docker `py_compile` passed for both runners and the
  figure generator. Smoke validation passed for OASIS multistage, LungCT IDIR,
  LungCT SINR global-NCC+FD-BE, LungCT DirectCP cps4 global-NCC+FD-BE, and
  LungCT DirectCP multistage global-NCC+FD-BE.
- Full launch completed `9/9` new jobs under
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1`; all launcher
  queue statuses were `0`.
- Full diagnostic CSV validation passed. New 2500-epoch runs wrote `251`
  global-NCC diagnostic rows; LungCT 2400-epoch multistage runs wrote `241`
  rows. All diagnostic losses were finite and covered epoch `1` through the
  final scheduled epoch.
- Final global NCC losses:
  - OASIS DirectCP 8-4-2 `-0.994035`.
  - LungCT case01: IDIR `-0.946411`, SINR cps4 `-0.997760`,
    DirectCP cps4 `-0.998162`, DirectCP 32-16-8-4 `-0.998160`.
  - LungCT case08: IDIR `-0.910503`, SINR cps4 `-0.993551`,
    DirectCP cps4 `-0.992049`, DirectCP 32-16-8-4 `-0.997240`.
- Standardized outputs written in
  `experiments/paper_artifacts/fitting_figure/v001/`:
  `fit_modes_standardized_global_ncc_panelA_curves.png`,
  `fit_modes_standardized_global_ncc_panelA_selected_runs.csv`,
  `fit_modes_standardized_global_ncc_panelA_curve_summary.csv`,
  `fit_modes_standardized_global_ncc_panelA_standard_validation.csv`,
  `fit_modes_standardized_global_ncc_panelA_standard_validation_report.md`,
  and `fit_modes_standardized_global_ncc_panelA_manifest.json`.
- 2026-07-01 update: kept the linear global-NCC-loss Panel A view with shared
  y-axis range `[-1.0, -0.8]` and added a second PNG,
  `fit_modes_standardized_global_ncc_panelA_1minusncc_log_curves.png`, plotting
  `1 - NCC` on a log y-axis.
- Visual check: Panel A y tick labels plus the y-axis label are visible on
  every subplot.
