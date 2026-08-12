# Stage 1 PL1C Default Report

## Status

Stage 1 completed successfully on 2026-08-11.

- Variant: `PL1C`.
- Loss branch: `ptv_lcc_isotv`.
- pTV isoTV weight: `0.055`.
- Schedule: original DUAL blur `[4,2,0]`.
- Cases: DIR-Lab 4DCT `01-10`.
- Accepted records: `10/10`.
- Posthoc regularity complete: `10/10`.
- Attempts: all cases completed on attempt 1.
- Output root:
  `<container-data-root>/inrorspline/runs/lungct_dual_inr_camera_ready_curve_stage1_pl1c_default_v1`.

Case `01` was first run as the practical smoke and then reused in the full
launcher. Cases `02-10` ran on all four GPUs.

## Aggregate Summary

| Method | Native snap TRE mean +/- SD (mm) | Reg-grid subvoxel TRE mean +/- SD (mm) | FG SDlogJ mean +/- SD | FG J<=0 pct mean +/- SD | Runtime mean +/- SD (s) |
|---|---:|---:|---:|---:|---:|
| pTVReg reference | 0.919 +/- 0.161 | n/a | n/a | n/a | n/a |
| MR-D-BSCP pTV-loss `w=0.055` | 0.917 +/- 0.156 | n/a | n/a | n/a | n/a |
| PL1A original blur/native loss | 0.976 +/- 0.175 | 1.114 +/- 0.128 | 0.1434 +/- 0.0374 | 0.0000 +/- 0.0000 | 359.2 +/- 10.8 |
| PL1C original blur/pTV loss `w=0.055` | 0.991 +/- 0.173 | 1.101 +/- 0.130 | 0.2099 +/- 0.0599 | 0.0001 +/- 0.0003 | 119.0 +/- 5.9 |

PL1C improves registration-grid subvoxel TRE relative to PL1A, but its native
snap TRE is slightly worse than PL1A and remains worse than the pTVReg and
MR-D-BSCP pTV-loss reference means.

## Case-by-case Native Snap TRE

Lower is better. Delta columns are PL1C minus the comparator, so positive
values mean PL1C has worse TRE.

| Case | pTVReg | MR-D-BSCP `w=0.055` | PL1A | PL1C | PL1C - pTVReg | PL1C - MR | PL1C - PL1A |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.767 | 0.793 | 0.753 | 0.770 | +0.003 | -0.022 | +0.017 |
| 02 | 0.751 | 0.776 | 0.708 | 0.768 | +0.018 | -0.008 | +0.060 |
| 03 | 0.927 | 0.948 | 0.896 | 0.910 | -0.017 | -0.038 | +0.014 |
| 04 | 1.266 | 1.250 | 1.289 | 1.343 | +0.077 | +0.093 | +0.054 |
| 05 | 1.078 | 1.090 | 1.184 | 1.141 | +0.063 | +0.051 | -0.044 |
| 06 | 0.832 | 0.830 | 0.975 | 0.967 | +0.136 | +0.137 | -0.007 |
| 07 | 0.806 | 0.796 | 0.943 | 0.947 | +0.141 | +0.151 | +0.004 |
| 08 | 1.014 | 1.001 | 1.042 | 1.108 | +0.093 | +0.107 | +0.065 |
| 09 | 0.910 | 0.882 | 1.007 | 0.997 | +0.087 | +0.115 | -0.010 |
| 10 | 0.841 | 0.809 | 0.960 | 0.958 | +0.117 | +0.150 | -0.002 |

Win counts on native snap TRE:

- PL1C beats pTVReg in `1/10` cases.
- PL1C beats MR-D-BSCP pTV-loss `w=0.055` in `3/10` cases.
- PL1C beats PL1A in `4/10` cases.

## PL1C Per-case Metrics

| Case | Native snap TRE (mm) | Reg-grid subvoxel TRE (mm) | FG SDlogJ | FG J<=0 pct | Total wall time (s) |
|---:|---:|---:|---:|---:|---:|
| 01 | 0.770 | 0.944 | 0.1419 | 0.0000 | 122.0 |
| 02 | 0.768 | 0.937 | 0.1026 | 0.0000 | 121.0 |
| 03 | 0.910 | 1.103 | 0.1436 | 0.0000 | 120.4 |
| 04 | 1.343 | 1.358 | 0.2572 | 0.0000 | 125.2 |
| 05 | 1.141 | 1.226 | 0.2510 | 0.0009 | 121.9 |
| 06 | 0.967 | 1.107 | 0.2618 | 0.0000 | 115.3 |
| 07 | 0.947 | 1.042 | 0.2227 | 0.0000 | 110.6 |
| 08 | 1.108 | 1.199 | 0.2039 | 0.0000 | 122.0 |
| 09 | 0.997 | 1.063 | 0.2716 | 0.0003 | 124.5 |
| 10 | 0.958 | 1.031 | 0.2427 | 0.0000 | 107.6 |

## Acceptance

The technical acceptance report passed:

- accepted records: `10/10`;
- regularity complete: `10/10`;
- all records use protocol `PL`;
- all records use fixed T00 1 mm lung mask for sampling and metric masks;
- semantic/lobe/segmentation loss is disabled;
- all records visited author blur sigmas `[4,2,0]`;
- all records have finite TRE and common regularity metrics.

TRE and regularity magnitude were not used as completion gates.

## Interpretation

The PL1C default point is technically valid and should be kept for the compact
curve sweep, but it is not an obvious final camera-ready point by itself:

- It does not close the aggregate native snap TRE gap to pTVReg or MR-D-BSCP.
- It is slightly worse than PL1A on native snap TRE.
- It is smoother than MR-D-BSCP only if judged by the existing PL regularity
  convention; its foreground stdlogJ is higher than PL1A.
- It is much faster than previous PL1A/PL1D/PL1B rows because PL1C samples 512
  local-CC centers per step instead of 30k global-NCC points.

The next useful step is still Stage 2: run the compact all-case sweep around
this point, especially PL1C weights `[0.0055,0.017,0.055,0.17,0.55]`. The
default `0.055` point alone does not justify adding PL1C to the camera-ready
figure, but the weight-response curve may still provide a useful tradeoff line.

## Source Files

- PL1C optimization index:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_optimization_index.csv`
- PL1C population statistics:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_population_statistics.csv`
- PL1C regularity:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_regularity_metrics.csv`
- PL1C acceptance:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/stage1_pl1c_default_acceptance_report.md`
