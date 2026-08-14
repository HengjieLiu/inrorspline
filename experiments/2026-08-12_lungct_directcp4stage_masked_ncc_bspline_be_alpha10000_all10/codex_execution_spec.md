---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-08-12
---

# LungCT Corrected MR-D-BSCP IDIR-Loss Test

## Goal

Run a corrected all-10 DIR-Lab 4DCT test for the `MR-D-BSCP (IDIR loss)` row.
This keeps the current best MR-D-BSCP schedule and alpha from the all-case
full-crop FD-BE sweep, but changes the optimization protocol to the intended
IDIR-style fixed T00 lung-mask sampled domain and analytic cubic B-spline
bending-energy regularizer.

## Configuration

```text
method = direct_cp_32_16_8_4_masked_ncc_bspline_be
cases = 01-10
alpha = 10000
lr = 1e-3
seed = 1
batch_size = 10000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
```

Training loss:

```text
loss = fixed_T00_lung_mask_sampled_global_NCC
     + alpha * sampled_analytic_cubic_bspline_BE
```

The images are the canonical pTVReg-style 1mm landmark-crop arrays, not
lung-masked images. The fixed T00 lung mask defines the sampled loss domain and
the foreground regularity metric domain.

## Comparison Target

Compare against the current plotted best:

```text
method = direct_cp_32_16_8_4_global_ncc_fd_be
loss = full-crop global NCC + dense forward-difference BE
mask = fixed T00 lung mask metric-only
alpha = 10000
TRE = 0.966 +/- 0.151 mm
foreground stdLogJ = 0.185452 +/- 0.060879
```

## Commands

Compile/smoke:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/common.py \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_best_all10.py \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_directcp4stage_masked_ncc_bspline_be_best_all10.py
```

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_32_16_8_4_masked_ncc_bspline_be \
  --case-id 1 \
  --alpha 10000 \
  --stage-epochs 1,1,1,1 \
  --out-root ${DATA_ROOT}/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10_smoke_v1 \
  --gpu 0 \
  --overwrite
```

Dry-run launch:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_best_all10.py \
  --dry-run
```

Full launch:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_best_all10.py \
  --gpus 0,1,2,3
```

Aggregation:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_directcp4stage_masked_ncc_bspline_be_best_all10.py
```

## Outputs

Run root:

```text
${DATA_ROOT}/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10_v1
```

Review root:

```text
experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha10000_all10/
```
