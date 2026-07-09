---
doc_type: technical_log
status: completed
date_started: 2026-07-07
date_completed: 2026-07-07
---

# Technical Log

## Known Limitations

- This is a COPD sanity run for one method only, not a full COPD Pareto panel.
- COPD masks are generated and propagated for QA and foreground regularity
  metrics only. The pTV-LCC + isoTV optimization uses the pTV border crop and
  does not use lung-mask loss weighting.
- The method uses the 4DCT-selected `ptv_isotv_weight=0.11` and the same
  multistage `32,16,8,4` workflow as the 4DCT run. COPD official pTVReg
  best/fine-tuned settings are closer to an `8 -> 4` refinement path, so this
  is a sanity check, not an official COPD pTVReg reproduction.
- COPD preprocessing uses the requested plan-compatible crop shape table. For
  case `01`, this gives pTV 1mm shape `[275,201,229]`; the exact MATLAB
  1-based crop formula would be one z-slice smaller at the clamped boundary.
  Each audit file records both the used crop bounds and the exact-formula
  bounds.
- Headline TRE is native-crop/original-domain snap-to-voxel TRE. Reg-1mm
  snap/subvoxel TREs are diagnostics.

## Verification Checklist

- Compile changed scripts with `python -m py_compile`.
- Generate COPD case `01` pTV-prep and verify landmarks are inside crop/reg
  grids and case `01` 1mm shape is `[275,201,229]`.
- Run case `01` training smoke with stage epochs `1,1,1,1`; require completed
  summary, finite TRE, finite SDlogJ, and saved dense DVF.
- Launch cases `01-10` on GPUs `0,1,2,3`; require `10/10` completed summaries.
- Aggregate `copd_case_rows.csv`, `copd_mean_row.csv`, and
  `review_report.md`.

## Verification Results

- Compile: passed in `py310sitreg_v1_test1`.
- COPD masks: generated for cases `01-10`, phases `iBH` and `eBH`, under
  `/database/datasets/lungCT/processed/masks/copd/caseXX`.
- COPD pTV-prep: generated for cases `01-10`; all verification flags passed.
  Case `01` smoke shape matched `[275,201,229]`.
- Loader audit: passed for all 10 COPD cases with non-empty fixed masks and
  finite identity TREs.
- Training smoke: case `01`, stage epochs `1,1,1,1`, completed with finite
  headline TRE, finite foreground SDlogJ `0.031219`, and saved dense DVFs.
- Full launch: completed `10/10` summaries with return code `0` on GPUs
  `0,1,2,3`.

## Full Run Results

Review artifacts:

```text
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/copd_case_rows.csv
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/copd_mean_row.csv
experiments/2026-07-07_copd_directcp32_ptvlcc_w0p11_sanity/review_report.md
```

Headline native-crop/original snap TRE:

| Case | TRE mm | SDlogJ |
|---:|---:|---:|
| 1 | 6.683 | 0.7532 |
| 2 | 5.304 | 0.7013 |
| 3 | 0.824 | 0.2755 |
| 4 | 16.298 | 0.9898 |
| 5 | 15.952 | 1.0254 |
| 6 | 9.044 | 0.8035 |
| 7 | 0.849 | 0.2586 |
| 8 | 5.070 | 0.5785 |
| 9 | 0.635 | 0.2384 |
| 10 | 7.825 | 0.6430 |

Mean row:

```text
native_crop_snap_tre_mean_mm = 6.848574702974491
native_crop_snap_tre_std_across_cases_mm = 5.426121090575178
native_crop_sub_tre_mean_mm = 6.916116421001098
reg1mm_snap_tre_mean_mm = 7.029276781887653
fg_log_jdet_std_mean = 0.6267116184356266
fg_jdet_le0_pct_mean = 4.7830936942178015
elapsed_sec_mean = 270.4311049699783
```

Interpretation: the 4DCT-selected `32-16-8-4, w=0.11` workflow is not robust on
COPD as-is. Cases `03`, `07`, and `09` are sub-millimeter, but cases `04`,
`05`, `06`, and `10` remain high-error and high-regularity/folding outliers.
Use this as evidence to run a COPD-tuned follow-up rather than as the final COPD
curve.

## Commands

Compile:

```bash
docker exec py310sitreg_v1_test1 python -m py_compile \
  /homebase/code_sync/inrorspline/scripts/data/common/lungct.py \
  /homebase/code_sync/inrorspline/scripts/data/preprocessing/lungct/generate_copd_lung_masks.py \
  /homebase/code_sync/inrorspline/scripts/data/preprocessing/lungct/generate_copd_ptvprep.py \
  /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/common.py \
  /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/run_one.py \
  /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_copd_directcp32_ptvlcc_w0p11_all10.py \
  /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/aggregate_copd_directcp32_ptvlcc_w0p11.py
```

Preprocess:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/data/preprocessing/lungct/generate_copd_lung_masks.py --case-ids 1-10 --data-root /database/datasets/lungCT --stage-root /database/datasets/lungCT/processed/masks --device cuda --gpu 0
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/data/preprocessing/lungct/generate_copd_ptvprep.py --case-ids 1-10 --data-root /database/datasets/lungCT --mask-root /database/datasets/lungCT/processed/masks --out-root /database/datasets/lungCT/processed/preprocessing/ptvprep/ptvreg_landmark_crop_1mm_trilinear_v1 --device cuda
```

Full launch:

```bash
docker exec py310sitreg_v1_test1 python /homebase/code_sync/inrorspline/scripts/experiments/lungct_no_sweep_repro/launch_copd_directcp32_ptvlcc_w0p11_all10.py --gpus 0,1,2,3
```
