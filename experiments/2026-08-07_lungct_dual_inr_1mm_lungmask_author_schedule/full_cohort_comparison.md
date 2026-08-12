# PL1A Full-Cohort Comparison

`PL1A` runs the same `PL` pTVReg 1 mm fixed T00 lung-mask protocol as
`PL1D/PL1B`, but uses the released DUAL-INR original blur schedule instead of
the controlled `[8,4,2,1]` image pyramid.

Technical completion passed with `10/10` accepted records and `10/10` posthoc
regularity records.

## Aggregate Summary

| Method | Native snap TRE mean +/- SD (mm) | Reg-grid subvoxel TRE mean +/- SD (mm) | FG SDlogJ mean +/- SD | FG J<=0 pct mean +/- SD | Runtime mean +/- SD (s) |
|---|---:|---:|---:|---:|---:|
| pTVReg reference | 0.919 +/- 0.161 | 1.057 +/- 0.123 | 0.1977 +/- 0.0604 | 0.0278 +/- 0.0338 | n/a |
| MR-D-BSCP pTV-loss `w=0.055` | 0.917 +/- 0.156 | 1.049 +/- 0.129 | 0.2238 +/- 0.0607 | 0.0367 +/- 0.0431 | 246.6 +/- 30.1 |
| PL1D image pyramid, native DUAL reg | 0.981 +/- 0.169 | 1.117 +/- 0.129 | 0.1417 +/- 0.0369 | 0.0000 +/- 0.0000 | 443.7 +/- 35.9 |
| PL1B image pyramid, IDIR-type BE | 0.986 +/- 0.169 | 1.107 +/- 0.130 | 0.1650 +/- 0.0431 | 0.0000 +/- 0.0002 | 638.4 +/- 38.1 |
| PL1A author blur, native DUAL reg | 0.976 +/- 0.175 | 1.114 +/- 0.128 | 0.1434 +/- 0.0374 | 0.0000 +/- 0.0000 | 359.2 +/- 10.8 |

## Case-by-case Native Snap TRE

Lower is better.

| Case | pTVReg | MR-D-BSCP `w=0.055` | PL1D | PL1B | PL1A |
|---:|---:|---:|---:|---:|---:|
| 01 | 0.767 | 0.793 | 0.765 | 0.821 | 0.753 |
| 02 | 0.751 | 0.776 | 0.747 | 0.766 | 0.708 |
| 03 | 0.927 | 0.948 | 0.877 | 0.939 | 0.896 |
| 04 | 1.266 | 1.250 | 1.274 | 1.344 | 1.289 |
| 05 | 1.078 | 1.090 | 1.187 | 1.157 | 1.184 |
| 06 | 0.832 | 0.830 | 0.995 | 0.996 | 0.975 |
| 07 | 0.806 | 0.796 | 0.940 | 0.918 | 0.943 |
| 08 | 1.014 | 1.001 | 1.092 | 1.075 | 1.042 |
| 09 | 0.910 | 0.882 | 0.987 | 0.952 | 1.007 |
| 10 | 0.841 | 0.809 | 0.950 | 0.897 | 0.960 |

## Readout

- PL1A beats pTVReg on native snap TRE in `3/10` cases and beats MR-D-BSCP
  pTV-loss `w=0.055` in `3/10` cases.
- PL1A beats PL1D on native snap TRE in `5/10` cases.
- PL1A improves the PL native-reg aggregate relative to PL1D
  (`0.9758` vs `0.9815` mm), but the improvement is small and does not close
  the gap to pTVReg or MR-D-BSCP.
- PL1A is faster than PL1D in this implementation (`359.2` vs `443.7` s mean
  total wall time), despite using `3000` steps, because it avoids rebuilding
  per-stage downsampled loss volumes and optimizer state.
- PL1A retains the main PL smoothness advantage: foreground `J<=0` is `0.0`
  across the cohort, and FG SDlogJ is lower than pTVReg/MR-D-BSCP.

## Realized Blur Schedule

All cases visited the original DUAL blur sigmas `[4,2,0]`.

| Blur sigma | Mean stage length +/- SD (steps) |
|---:|---:|
| 4 | 362.7 +/- 70.7 |
| 2 | 215.4 +/- 27.4 |
| 0 | 2421.9 +/- 80.9 |

## Source Files

- PL1A:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_author_schedule/optimization_index.csv`
- PL1D/PL1B:
  `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/optimization_index.csv`
- pTVReg:
  `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv`
- MR-D-BSCP pTV-loss `w=0.055`:
  `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_case_rows.csv`
