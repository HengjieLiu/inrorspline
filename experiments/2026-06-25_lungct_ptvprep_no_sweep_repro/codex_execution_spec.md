---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-06-25
---

# LungCT pTV-Prep No-Sweep Reproduction

## Goal

Run a fixed-parameter 4DCT LungCT reproduction pass on the canonical cropped
1mm pTV-preprocessed images, without alpha or isoTV-weight sweeps.

## Inputs

- Dataset: DIR-Lab 4DCT cases `01-10`, fixed `T00`, moving `T50`.
- Canonical pTV-prep root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`
- Per-case arrays:
  - `fixed_T00_reg1mm_trilinear_zyx.npy`
  - `moving_T50_reg1mm_trilinear_zyx.npy`
  - `fixed_T00_lungmask_reg1mm_nearest_zyx.npy`
  - `landmarks_fixed_T00_reg1mm_zyx.npy`
  - `landmarks_moving_T50_reg1mm_zyx.npy`

## Methods

- `idir_be_default_lungmask`: sampled NCC plus IDIR bending, fixed T00 lung-mask
  training samples, `alpha_bending=10`, `lr=1e-4`, `epochs=2500`.
- `sinr_cps4_ptv_lcc_isotv`: SIREN-generated cps4 cubic B-spline controls,
  pTV LCC plus isoTV, `lr=1e-4`, `epochs=2500`.
- `direct_cp_cps4_ptv_lcc_isotv`: direct cps4 cubic B-spline controls, pTV LCC
  plus isoTV, `lr=1e-3`, `epochs=2500`.
- `direct_cp_32_16_8_4_ptv_lcc_isotv`: direct multistage controls with exact
  cubic dyadic knot insertion, full CPS `32,16,8,4`, downsample `8,4,2,1`,
  stage epochs `200,400,800,1000`, `lr=1e-3`.

Shared pTV settings for non-IDIR methods:

```text
ptv_lcc_sigma_mm=2.1
ptv_isotv_weight=0.11
ptv_csqrt=5e-3
border_mask_voxels=5
```

## Execution

Use the default Docker container:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_all10_4gpu.py --gpus 0,1,2,3
```

Large run output root:

```text
/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1
```

Aggregate after completion:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/aggregate_report.py
```

## Reporting

The report must include original/native-crop identity/snap/subvoxel TRE,
cropped-1mm identity/snap/subvoxel TRE, fixed-lung-mask foreground
log-Jacobian std, folding percentage, bending energy, diffusion energy,
displacement magnitude statistics, runtime, and peak memory.

Use original/native-crop snap TRE as the pTVReg-comparable headline metric.
Cropped-1mm TRE is a diagnostic because transformed landmarks are fractional in
the resampled grid.

Interpret the new rows as canonical pTV-prep no-sweep runs. The multistage
schedule differs from the legacy full task4 row `300,500,800,1200` and the
legacy reduced row `200,200,200,400`, so those references are anchors, not
byte-identical reproduction targets.
