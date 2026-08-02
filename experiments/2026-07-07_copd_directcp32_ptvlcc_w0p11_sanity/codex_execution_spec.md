---
doc_type: codex_execution_spec
status: completed
date_planned: 2026-07-07
date_completed: 2026-07-07
---

# COPD DirectCP32 pTV-LCC + isoTV Sanity Run

> Historical paths are preserved as executed. Current equivalent:
> `/database/datasets/lungCT/DIR-Lab/processed/pTVReg_iso1mm`.

## Goal

Run a COPD cases `01-10` sanity pass for
`direct_cp_32_16_8_4_ptv_lcc_isotv`, reported as
`MR-D-BSCP (pTVReg loss)`, using the same pTV-style image preprocessing and
optimization settings as the 4DCT no-sweep run where applicable.

## Inputs

- Dataset: COPD cases `01-10`.
- Fixed image: `iBHCT`.
- Moving image: `eBHCT`.
- Direction: `eBHCT -> iBHCT`.
- Canonical pTV-prep root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`
- COPD pTV-prep case root:
  `.../copd/case01..case10`
- Landmark convention: source `xyz`, pTVReg/MATLAB 1-based voxel coordinates;
  saved as `zyx`, Python 0-based voxel coordinates.

## Preprocessing

Generate pTV-style COPD preprocessing:

```text
intensity: clip [80,900], subtract 80, divide by 820
crop: landmark union with margin xyz [10,10,5]
registration grid: round(crop_shape_zyx * native_spacing_zyx)
image resampling: trilinear, align_corners=True
mask resampling: nearest-neighbor, QA/metric use only
```

Expected case `01` pTV 1mm shape: `[275,201,229]`.

## Method

```text
method: direct_cp_32_16_8_4_ptv_lcc_isotv
display: MR-D-BSCP (pTVReg loss)
stage_full_cps: 32,16,8,4
stage_downsample: 8,4,2,1
stage_epochs: 200,400,800,1000
lr: 1e-3
seed: 1
ptv_lcc_sigma_mm: 2.1
ptv_isotv_weight: 0.11
ptv_csqrt: 5e-3
border_mask_voxels: 5
save_dense_dvf: true
```

## Execution

Use the default Docker container:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_copd_directcp32_ptvlcc_w0p11_all10.py --gpus 0,1,2,3
```

Large run output root:

```text
/database/inrorspline/runs/copd_directcp32_ptvlcc_w0p11_sanity_v1
```

Aggregate after completion:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/aggregate_copd_directcp32_ptvlcc_w0p11.py
```

## Reporting

Primary output files:

```text
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/copd_case_rows.csv
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/copd_mean_row.csv
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/review_report.md
```

Use native-crop/original-domain snap-to-voxel TRE as the headline metric.
Cropped-1mm TRE is retained as a diagnostic because landmarks are integer in
native/crop coordinates and fractional after 1mm resampling.
