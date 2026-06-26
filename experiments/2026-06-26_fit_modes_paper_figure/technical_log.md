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
