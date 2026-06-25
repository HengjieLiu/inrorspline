---
doc_type: codex_execution_spec
status: planned
date_planned: 2026-06-25
---

# LungCT IDIR + DirectCP 32-16-8-4 Regularization Sweep

## Goal

Run a coarse regularization sweep for the two LungCT methods that are in the
best current range: IDIR BE default and DirectCP `32,16,8,4` pTV LCC + isoTV.

## Sweep Grid

Reuse the completed no-sweep `1e0` baseline from:

```text
/database/inrorspline/runs/lungct_ptvprep_no_sweep_repro_v1
```

Launch only nonbaseline factors `0.01,0.1,10,100`.

IDIR:

```text
base alpha_bending = 10
values = 0.1, 1, 100, 1000
```

DirectCP:

```text
base ptv_isotv_weight = 0.11
values = 0.0011, 0.011, 1.1, 11
```

Fixed settings:

```text
cases = 01-10
seed = 1
IDIR lr = 1e-4
IDIR epochs = 2500
DirectCP lr = 1e-3
DirectCP stage_epochs = 200,400,800,1000
DirectCP stage schedule = 32,16,8,4
DirectCP downsample schedule = 8,4,2,1
ptv_lcc_sigma_mm = 2.1
ptv_csqrt = 5e-3
border_mask_voxels = 5
```

## Outputs

Large run root:

```text
/database/inrorspline/runs/lungct_ptvprep_reg_sweep_idir_directcp32_v1
```

Review root:

```text
experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/
```

The report must keep all four TREs:

- `original_snap`: primary comparison metric.
- `original_subvoxel`.
- `reg1mm_snap`.
- `reg1mm_subvoxel`.

Also report fixed T00 lung-mask foreground regularity metrics and Pareto plots
against `original_snap`.

