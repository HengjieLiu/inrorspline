---
doc_type: codex_execution_spec
status: completed
date_planned: 2026-06-25
---

# LungCT DirectCP32 Global NCC + Dense FD-BE Alpha Sweep

> Historical paths are preserved as executed. Current equivalent:
> `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.

## Goal

Run a selected-case diagnostic for DirectCP multistage `32,16,8,4` on canonical
cropped 1mm pTV-prep 4DCT inputs, replacing the pTV LCC + isoTV objective with
SINR-style full-crop global NCC plus dense forward-difference bending energy.

This is the practical follow-up to the analytic cubic B-spline BE diagnostic:
the exact analytic training loss was implemented and smoked, but its runtime was
too high for this tuning pass.

## Configuration

```text
method = direct_cp_32_16_8_4_global_ncc_fd_be
cases = 01,08
alphas = 0,0.1,1,10,100,1000,10000,100000,1000000
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
seed = 1
gpus = 2,3
```

Training loss:

```text
loss = full_crop_global_NCC + alpha * dense_forward_difference_BE
```

The training loss uses no lung-mask or border-mask weighting. The fixed T00 lung
mask is used only for foreground regularity metrics.

## Inputs And Outputs

Input root:

```text
/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1
```

Run root:

```text
/database/inrorspline/runs/lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep_v1
```

Review root:

```text
experiments/2026-06-25_lungct_directcp32_global_ncc_fdbe_case01_case08_alpha_sweep/
```

## Commands

Dry run:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_fdbe_sweep.py \
  --cases 1,8 \
  --alphas 0,0.1,1,10,100,1000,10000,100000,1000000 \
  --gpus 2,3 \
  --dry-run
```

Full launch:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_directcp32_global_ncc_fdbe_sweep.py \
  --cases 1,8 \
  --alphas 0,0.1,1,10,100,1000,10000,100000,1000000 \
  --gpus 2,3
```

Aggregation:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/aggregate_directcp32_global_ncc_fdbe_sweep.py
```

## Reporting

Use `original_snap` TRE as the primary case-wise value. Also retain
`original_subvoxel`, `reg1mm_snap`, and `reg1mm_subvoxel` TRE.

Report fixed-T00-lung-mask foreground regularity: stdLogJ, `J<=0%`, finite
difference BE/diffusion, displacement mean/p95/max, elapsed seconds, and peak
GPU memory.
