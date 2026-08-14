---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-08-12
---

# LungCT Corrected MR-D-BSCP IDIR-Loss Alpha Sweep

## Goal

Sweep the corrected `MR-D-BSCP (IDIR loss)` protocol over a wider alpha range
after the single `alpha=10000` operating point showed a likely regularization
scale mismatch.

This keeps the corrected fixed T00 lung-mask sampled image-loss domain and the
sampled analytic cubic B-spline bending-energy regularizer.

## Configuration

```text
method = direct_cp_32_16_8_4_masked_ncc_bspline_be
cases = 01-10
new_alphas = 100,1000,100000,1000000
anchor_alpha = 10000
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

## Commands

Compile:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py
```

Dry-run:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py \
  --dry-run
```

Full launch:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/launch_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py \
  --gpus 0,1,2,3
```

Aggregation:

```bash
docker exec py310sitreg_v1_test1 python \
  ${REPO_ROOT}/scripts/experiments/lungct_no_sweep_repro/aggregate_directcp4stage_masked_ncc_bspline_be_alpha_sweep.py
```

## Outputs

Run root:

```text
${DATA_ROOT}/runs/lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep_v1
```

Review root:

```text
experiments/2026-08-12_lungct_directcp4stage_masked_ncc_bspline_be_alpha_sweep/
```
