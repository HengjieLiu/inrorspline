---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-07-07
---

# COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV Sanity Run

## Goal

Run COPD cases `01-10` for the same `MR-D-BSCP (pTVReg loss)` settings as the
prior COPD sanity run, changing only the multistage pyramid to add a coarser
64mm-equivalent initialization stage while keeping total epochs at `2500`.

## Inputs

- Dataset: COPD cases `01-10`.
- Fixed image: `iBHCT`.
- Moving image: `eBHCT`.
- Direction: `eBHCT -> iBHCT`.
- pTV-prep root:
  `/database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1`

## Method

```text
method: direct_cp_32_16_8_4_ptv_lcc_isotv
display: MR-D-BSCP 64-32-16-8-4 (pTVReg loss)
stage_full_cps: 64,32,16,8,4
stage_downsample: 16,8,4,2,1
stage_epochs: 100,200,400,800,1000
total_epochs: 2500
lr: 1e-3
seed: 1
ptv_lcc_sigma_mm: 2.1
ptv_isotv_weight: 0.11
ptv_csqrt: 5e-3
border_mask_voxels: 5
save_dense_dvf: true
```

## Execution

Large output root:

```text
/database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1
```

Launch:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_copd_directcp32_ptvlcc_w0p11_all10.py \
  --gpus 0,1,2,3 \
  --out-root /database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1 \
  --log-root /homebase/code_sync/inrorspline/logs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1 \
  --stage-full-cps 64,32,16,8,4 \
  --stage-downsample 16,8,4,2,1 \
  --stage-epochs 100,200,400,800,1000
```

Aggregate:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/aggregate_copd_directcp32_ptvlcc_w0p11.py \
  --out-root /database/inrorspline/runs/copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity_v1 \
  --review-root /homebase/code_sync/inrorspline/experiments/2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_sanity \
  --title "COPD DirectCP64-32-16-8-4 pTV-LCC + isoTV Sanity Run" \
  --display-name "MR-D-BSCP 64-32-16-8-4 (pTVReg loss)"
```

## Reporting

Use native-crop/original-domain snap-to-voxel TRE as the headline metric.
Compare directly against the prior `32,16,8,4` COPD sanity run to decide whether
the coarser stage helped the large-error cases.
