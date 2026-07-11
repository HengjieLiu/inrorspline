---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-06-26
---

# LungCT Case08 DirectCP 4-Stage BE Implementation Comparison

## Goal

Compare two training regularizers for the same DirectCP 4-stage LungCT case `08`
run:

- full-crop global NCC + dense forward-difference BE
- full-crop global NCC + analytic cubic B-spline BE

## Configuration

```text
case = 08
stage_full_cps = 32,16,8,4
stage_downsample = 8,4,2,1
stage_epochs = 200,400,800,1000
lr = 1e-3
alpha = 10000
seed = 1
```

Use the fixed T00 lung mask only for foreground regularity metrics, not for the
training loss.

## Inputs And Outputs

Dataset root:

```text
/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1
```

Output root:

```text
/database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1
```

Review root:

```text
experiments/2026-06-26_lungct_case08_be_impl_compare_alpha1e4/
```

## Commands

FD-BE full run:

```bash
docker exec -w /homebase/code_sync/inrorspline py310sitreg_v1_test1 \
  python scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_32_16_8_4_global_ncc_fd_be \
  --case-id 8 \
  --out-root /database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1 \
  --alpha 10000 \
  --lr 1e-3 \
  --stage-epochs 200,400,800,1000 \
  --gpu 2
```

Analytic B-spline BE full run:

```bash
docker exec -w /homebase/code_sync/inrorspline py310sitreg_v1_test1 \
  python scripts/experiments/lungct_no_sweep_repro/run_one.py \
  --method direct_cp_32_16_8_4_global_ncc_bspline_be \
  --case-id 8 \
  --out-root /database/inrorspline/runs/lungct_case08_be_impl_compare_alpha1e4_v1 \
  --alpha 10000 \
  --lr 1e-3 \
  --stage-epochs 200,400,800,1000 \
  --gpu 3
```

## Reporting

Report all four TRE metrics, with `original_snap` as the main comparison value.
Also report foreground regularity on the fixed T00 lung mask, optimization
elapsed seconds, and peak GPU memory.
