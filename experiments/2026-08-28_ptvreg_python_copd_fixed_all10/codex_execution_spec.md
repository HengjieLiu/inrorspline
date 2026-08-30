---
doc_type: codex_execution_spec
status: active_draft
---

# ptvreg-python COPD Fixed All10 Execution Spec

## Objective

Run the public `ptvreg-python` implementation on the COPDGene/DIR-Lab COPD
cases using fixed pTVReg-like options, then compare the result in the existing
LungCT TRE and dense-field regularity conventions.

## External

- Repository: `https://github.com/oazeybekoglu/ptvreg-python`
- Local path: `external/ptvreg-python`
- Import path: `external/ptvreg-python/src`
- API: `ptv.register_min.register_pair_min(moving, fixed, opts)`
- External code policy: do not modify external code for this experiment.

## Dataset And Protocols

- Dataset: COPD cases `01-10`.
- Canonical input root: `${LUNGCT_PTVREG_ISO1MM}/COPD`.
- Direction: moving `eBHCT` to fixed `iBHCT`.
- First protocol tag:
  `ptvreg_python_admm+canonical_ptv_landmark_crop_1mm_no_lungmask`.
- Second protocol tag:
  `ptvreg_python_admm+paper_hu_landmark_crop_1mm_no_lungmask`.
- Native-gate protocol tag:
  `ptvreg_python_admm+paper_hu_native_landmark_crop_no_lungmask`.
- The paper-HU variant reuses canonical pTVReg crop bounds and landmarks,
  clips raw HU to `[-1000, 500]`, normalizes to `[0, 1]`, and resamples to
  `round(crop_shape_zyx * native_spacing_zyx)` with PyTorch trilinear
  `align_corners=True`.
- The native-gate paper-HU variant loads raw COPD directly, uses the pTVReg
  landmark-bbox crop with margin `[x,y,z]=[10,10,5]`, clips raw HU to
  `[-1000, 500]`, normalizes to `[0,1]`, keeps the cropped array at native COPD
  spacing, and uses native R231 masks only for posthoc regularity.

## Fixed Options

```text
opt_method=admm
grid_spacing=(8,8,8)
isoTV=0.12
metric=lcc
metric_param=(2.1,2.1,2.1)
scale_metric_param=True
pix_resolution=(1,1,1)
spline_order=1
k_down=0.7
border_mask=5
interp_type=1
dtype=float64
```

For `paper_hu_native`, `pix_resolution` is overridden per case from raw COPD
native spacing so that `metric_param=(2.1,2.1,2.1)` remains a physical-mm LCC
width on the anisotropic grid.

No metric-based tuning, early stopping, or case-specific parameter changes are
allowed for the production all-10 run.

## Commands

Preflight tests:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python -m pytest scripts/experiments/ptvreg_python_repro/tests'
```

Canonical loader audit:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/run_copd_case.py --case-id 1 --audit-loader --audit-cases 1,2,3,4,5,6,7,8,9,10'
```

Smoke:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/run_copd_case.py --case-id 1 --synthetic-smoke --gpu 0 --synthetic-smoke-output experiments/2026-08-28_ptvreg_python_copd_fixed_all10/synthetic_gpu_smoke.json'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol canonical_1mm --smoke --gpus 0 --force'
```

Canonical all-10:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol canonical_1mm --gpus 0,1,2,3'
```

Paper-HU all-10:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol paper_hu_iso1mm --gpus 0,1,2,3'
```

Native paper-HU case08 gate:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/run_copd_case.py --protocol paper_hu_native --case-id 8 --prepare-paper-hu-only --prepare-cases 8 --gpu 0 --prepare-output experiments/2026-08-28_ptvreg_python_copd_fixed_all10/paper_hu_native_case08_input_manifest.json'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/run_copd_case.py --protocol paper_hu_native --case-id 8 --gpu 0 --overwrite'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/aggregate_copd_fixed_all10.py --protocol paper_hu_native --case-ids 8'
```

Native paper-HU all-10 dry-run/production:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol paper_hu_native --gpus 0,1,2,3 --dry-run'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/launch_copd_fixed_all10.py --protocol paper_hu_native --gpus 0,1,2,3'
```

Aggregate either protocol:

```bash
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/aggregate_copd_fixed_all10.py --protocol canonical_1mm'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/aggregate_copd_fixed_all10.py --protocol paper_hu_iso1mm'
docker exec py310sitreg_v1_test1 bash -lc 'cd /homebase/code_sync/inrorspline && python scripts/experiments/ptvreg_python_repro/aggregate_copd_fixed_all10.py --protocol paper_hu_native'
```

## Outputs

- Large run root:
  `/database/inrorspline/runs/ptvreg_python_copd_fixed_all10_v1`.
- Review root:
  `experiments/2026-08-28_ptvreg_python_copd_fixed_all10/`.
- Per-case output includes `summary.json`, `disp_vox_reg1mm_zyx.npy`,
  `disp_norm_xyz_reg1mm.npy`, `per_landmark_tre.csv`,
  `per_landmark_tre_native_crop.csv`, and `ptvreg_python_knots.pt`.
- Native-crop output writes `disp_vox_native_crop_zyx.npy` and
  `disp_norm_xyz_native_crop.npy` instead of the `reg1mm` dense-field names.
- Aggregation writes per-case CSV/JSON, mean-row CSV/JSON, and acceptance
  reports under the review root.

## Acceptance

- `10/10` cases completed for each launched production protocol.
- Four TRE conventions present:
  `native_crop_tre_snap_to_voxel`, `native_crop_tre_subvoxel`,
  `tre_snap_to_voxel`, and `tre_subvoxel`.
- Headline metric: `native_crop_tre_snap_to_voxel` in mm.
- Dense field shape equals each registration-grid input shape.
- No NaNs or infinities in saved dense fields.
- Dense-field foreground regularity metrics are finite.
- No production launch if conversion/import tests fail.
