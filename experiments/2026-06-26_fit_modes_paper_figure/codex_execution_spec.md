# Fit-Modes Paper Figure IDIR Eval Curves Execution Spec

## Objective

Add the missing IDIR matched image-similarity curves for the paper figure
without changing accepted training objectives or overwriting existing OASIS and
LungCT fit-mode results.

## Runs

- OASIS rank `9`, original sorted index `88`, fixed/moving `(0173,0136)`:
  IDIR original, alpha `1`, lr `1e-4`, 2500 epochs, GPU `0`.
- LungCT case `01`: IDIR pTV-preprocessed fixed-T00 lung-mask sampling,
  `alpha_bending=10`, lr `1e-4`, 2500 epochs, GPU `1`.
- LungCT case `08`: same LungCT IDIR protocol, `alpha_bending=10`, lr `1e-4`,
  2500 epochs, GPU `2`.

## Matched Similarity Eval

- OASIS writes `image_similarity_eval.csv` with full-image global NCC at epochs
  `1,10,20,...,2500`.
- LungCT writes `image_similarity_eval.csv` with pTV LCC data loss at epochs
  `1,10,20,...,2500`, using the same pTV LCC fixed cache and 5-voxel border
  mask convention as the SINR/DirectCP pTV LCC runs.
- Eval rows are posthoc diagnostics only: `affects_training=false`.
- IDIR training remains unchanged: sampled NCC plus IDIR bending energy.

## Paths

- Output root:
  `/database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1`
- Review root:
  `experiments/2026-06-26_fit_modes_paper_figure/`
- Existing source folders for later notebook:
  - `experiments/2026-06-26_oasis_rank009_sitreg_fit_modes_lr_sweep/`
  - `experiments/2026-06-26_lungct_case01_ptvreg_fit_modes_lr_sweep/`
  - `experiments/2026-06-26_lungct_case08_ptvreg_fit_modes_lr_sweep/`

## Launch Commands

Run inside Docker container `py310sitreg_v1_test1` from
`/homebase/code_sync/inrorspline`. Launch the three commands concurrently.
Use `python -u` so milestone eval rows are visible in logs during monitoring.

```bash
python -u scripts/experiments/oasis_alpha_sweep/run_one.py \
  --method idir_original \
  --alpha 1 \
  --lr 1e-4 \
  --gpu 0 \
  --case-rank 9 \
  --epochs 2500 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --variant fit_modes_paper_curves_idir_eval_v1 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/oasis_rank009 \
  --expected-original-case-index 88 \
  --expected-fixed-id 0173 \
  --expected-moving-id 0136 \
  --run-name oasis_rank009_idir_a1_lr1e-4_global_ncc_eval
```

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method idir_be_default_lungmask \
  --case-id 1 \
  --gpu 1 \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha-bending 10 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case01 \
  --run-name lungct_case01_idir_lr1e-4_ptv_lcc_eval
```

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method idir_be_default_lungmask \
  --case-id 8 \
  --gpu 2 \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha-bending 10 \
  --eval-interval 250 \
  --image-sim-eval-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_idir_eval_v1/lungct_case08 \
  --run-name lungct_case08_idir_lr1e-4_ptv_lcc_eval
```

## Acceptance

- Compile modified OASIS and LungCT runners inside Docker.
- Run a 20-epoch smoke with `--image-sim-eval-interval 10` and separate smoke
  output root.
- Full run produces 3 completed summaries and 3 `image_similarity_eval.csv`
  files.
- Each eval CSV has rows for epoch `1`, every tenth epoch, and epoch `2500`.
- Existing accepted fit-mode roots are not overwritten.

## Addendum: rRMSE LR Trajectory DirectCP 4-Stage Field Diagnostics

### Objective

Add the missing LungCT DirectCP 4-stage finest-domain target-field MSE
diagnostics needed for the rRMSE learning-rate trajectory figure. Existing
single-stage field-fit histories and the existing DirectCP 4-stage `lr=1e-2`
diagnostic are reused.

### Run Matrix

- LungCT case `01`, DirectCP 4-stage pTVReg-field fit, lr `1e-3`, GPU `0`.
- LungCT case `01`, DirectCP 4-stage pTVReg-field fit, lr `1e-4`, GPU `1`.
- LungCT case `08`, DirectCP 4-stage pTVReg-field fit, lr `1e-3`, GPU `2`.
- LungCT case `08`, DirectCP 4-stage pTVReg-field fit, lr `1e-4`, GPU `3`.

All runs use 2500 epochs, seed `1`, schedule `32,16,8,4`, downsample
`8,4,2,1`, stage epochs `400,400,800,900`, posthoc finest-domain MSE
diagnostics every 50 epochs during coarse stages and every epoch during the
final 1mm stage.

### Command Template

Run inside Docker container `py310sitreg_v1_test1` from
`/homebase/code_sync/inrorspline`:

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_case01_ptvreg_field_fit_one.py \
  --method direct_cp_4stage \
  --case-id CASE_ID \
  --gpu GPU_ID \
  --epochs 2500 \
  --lr LR \
  --stage-full-cps 32,16,8,4 \
  --stage-downsample 8,4,2,1 \
  --stage-epochs 400,400,800,900 \
  --finest-mse-diagnostic \
  --finest-mse-every 50 \
  --no-save-dense-dvf \
  --out-root /database/inrorspline/runs/fit_modes_paper_curves_directcp4stage_field_mse_diag_v1/lungct_caseCASE_ID \
  --run-name ptvreg_field_fit_direct_cp_4stage_caseCASE_ID_finestmsediag_sched32-16-8-4_e400-400-800-900_lrLR_TAG
```

### Acceptance

- Two-epoch smoke writes `stage_history.csv` with finite `finest_data_loss` and
  `finest_field_mse_objective_normalized_grid`.
- Each full run writes `2500` history rows and `935` finite finest-domain
  diagnostic points.
- Final finest-domain MSE values are finite and close to the corresponding
  original sweep final MSE summaries.
- The notebook writes `fit_modes_rrmse_lr_trajectory_curves.{png,pdf}` and the
  companion selected-run and curve-summary CSVs without overwriting the
  existing full comparison figure.

## Addendum: Panel A Global-NCC Curve Refresh

### Objective

Create a new Panel A curve package where OASIS rank9 and LungCT case01/case08
all use full-image or full-crop global NCC loss on the y-axis. IDIR keeps its
own training objective and logs global NCC as a diagnostic every 10 epochs.
Spline and DirectCP LungCT runs use full-crop global NCC + dense FD-BE for
training so the plotted loss is the optimized data term.

### Run Matrix

- OASIS rank `9`, DirectCP 8-4-2 multistage:
  alpha `1000`, lr `1e-3`, cps `8,4,2`, downsample `4,2,1`, stage epochs
  `400,800,1300`, total `2500`.
- LungCT case `01` and `08`, IDIR:
  `alpha_bending=3`, lr `1e-4`, `2500` epochs, native IDIR sampled NCC + BE,
  full-crop global NCC diagnostic every `10` epochs.
- LungCT case `01` and `08`, SINR cps4:
  alpha `10000`, lr `1e-4`, `2500` epochs, full-crop global NCC + dense
  forward-difference BE.
- LungCT case `01` and `08`, DirectCP cps4:
  alpha `10000`, lr `1e-3`, `2500` epochs, full-crop global NCC + dense
  forward-difference BE.
- LungCT case `01` and `08`, DirectCP 32-16-8-4 multistage:
  alpha `10000`, lr `1e-3`, cps `32,16,8,4`, downsample `8,4,2,1`, stage
  epochs `200,400,800,1000`, total `2400`, full-crop global NCC + dense
  forward-difference BE.

Existing OASIS Panel A IDIR/SINR cps4/DirectCP cps4 histories are reused from
the completed matched-eval and OASIS lr-sweep artifacts.

### Output Paths

- Full output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_v1`
- Smoke output root:
  `/database/inrorspline/runs/fit_modes_panelA_global_ncc_smoke_v1`
- Paper artifact output root:
  `experiments/paper_artifacts/fitting_figure/v001/`

### Command Templates

Run inside Docker container `py310sitreg_v1_test1` from
`/homebase/code_sync/inrorspline`.

OASIS DirectCP 8-4-2:

```bash
python -u scripts/experiments/oasis_alpha_sweep/run_one.py \
  --method direct_cp_8_4_2_be_bspline_analytic_multistage \
  --alpha 1000 \
  --lr 1e-3 \
  --gpu GPU_ID \
  --case-rank 9 \
  --epochs 2500 \
  --eval-interval 250 \
  --stage-full-cps 8,4,2 \
  --stage-downsample 4,2,1 \
  --stage-epochs 400,800,1300 \
  --global-ncc-diagnostic-interval 10 \
  --variant fit_modes_panelA_global_ncc_v1 \
  --out-root /database/inrorspline/runs/fit_modes_panelA_global_ncc_v1/oasis_rank009 \
  --expected-original-case-index 88 \
  --expected-fixed-id 0173 \
  --expected-moving-id 0136 \
  --run-name panelA_oasis_rank009_directcp842_sched8-4-2_ds4-2-1_e400-800-1300_a1000_lr0p001
```

LungCT IDIR:

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method idir_be_default_lungmask \
  --case-id CASE_ID \
  --gpu GPU_ID \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha-bending 3 \
  --eval-interval 250 \
  --global-ncc-diagnostic-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_panelA_global_ncc_v1/lungct_caseCASE_TAG \
  --run-name panelA_lungct_caseCASE_TAG_idir_a3_lr0p0001
```

LungCT SINR cps4:

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method sinr_cps4_global_ncc_fd_be \
  --case-id CASE_ID \
  --gpu GPU_ID \
  --epochs 2500 \
  --lr 1e-4 \
  --alpha 10000 \
  --eval-interval 250 \
  --global-ncc-diagnostic-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_panelA_global_ncc_v1/lungct_caseCASE_TAG \
  --run-name panelA_lungct_caseCASE_TAG_sinr_cps4_globalncc_fdbe_a10000_lr0p0001
```

LungCT DirectCP cps4:

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_cps4_global_ncc_fd_be \
  --case-id CASE_ID \
  --gpu GPU_ID \
  --epochs 2500 \
  --lr 1e-3 \
  --alpha 10000 \
  --eval-interval 250 \
  --global-ncc-diagnostic-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_panelA_global_ncc_v1/lungct_caseCASE_TAG \
  --run-name panelA_lungct_caseCASE_TAG_directcp_cps4_globalncc_fdbe_a10000_lr0p001
```

LungCT DirectCP 32-16-8-4:

```bash
python -u scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_32_16_8_4_global_ncc_fd_be \
  --case-id CASE_ID \
  --gpu GPU_ID \
  --lr 1e-3 \
  --alpha 10000 \
  --stage-full-cps 32,16,8,4 \
  --stage-downsample 8,4,2,1 \
  --stage-epochs 200,400,800,1000 \
  --eval-interval 250 \
  --global-ncc-diagnostic-interval 10 \
  --out-root /database/inrorspline/runs/fit_modes_panelA_global_ncc_v1/lungct_caseCASE_TAG \
  --run-name panelA_lungct_caseCASE_TAG_directcp321684_globalncc_fdbe_sched32-16-8-4_ds8-4-2-1_e200-400-800-1000_a10000_lr0p001
```

### Acceptance

- OASIS and LungCT runners pass Docker `py_compile`.
- Smoke jobs write finite `global_ncc_diagnostic.csv` rows.
- Full runs write expected diagnostic rows: `251` rows for 2500-epoch runs
  and `241` rows for 2400-epoch LungCT multistage runs.
- The standardized Panel A PNG has one shared y-axis range across all three
  cases and visible y tick labels/y-axis labels on every subplot.
