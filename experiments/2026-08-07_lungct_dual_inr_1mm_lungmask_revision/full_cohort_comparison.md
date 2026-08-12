# Full-Cohort PL1D / PL1B TRE Comparison

Full ten-case `PL1D`/`PL1B` aggregation completed with `20/20` accepted records and `20/20` posthoc regularity records. Cases `01` and `08` reuse the completed pilot records; all other records completed on attempt 1 in the full launcher.

## Aggregate Summary

| Method | Native snap TRE mean +/- SD (mm) | Reg-grid subvoxel TRE mean +/- SD (mm) | FG SDlogJ mean +/- SD | FG J<=0 pct mean +/- SD | Runtime mean +/- SD (s) |
|---|---:|---:|---:|---:|---:|
| pTVReg reference | 0.919 +/- 0.161 | 1.057 +/- 0.123 | 0.1977 +/- 0.0604 | 0.0278 +/- 0.0338 | n/a |
| MR-D-BSCP pTV-loss `w=0.055` | 0.917 +/- 0.156 | 1.049 +/- 0.129 | 0.2238 +/- 0.0607 | 0.0367 +/- 0.0431 | 246.6 +/- 30.1 |
| PL1D `dual_native_reg` | 0.981 +/- 0.169 | 1.117 +/- 0.129 | 0.1417 +/- 0.0369 | 0.0000 +/- 0.0000 | 443.7 +/- 35.9 |
| PL1B `idir_type_be` | 0.986 +/- 0.169 | 1.107 +/- 0.130 | 0.1650 +/- 0.0431 | 0.0000 +/- 0.0002 | 638.4 +/- 38.1 |

## Case-by-case Native Snap TRE

Lower is better. Delta columns are PL minus reference, so positive values mean PL is worse for TRE.

| Case | pTVReg | MR-D-BSCP `w=0.055` | PL1D | PL1D - pTVReg | PL1D - MR | PL1B | PL1B - pTVReg | PL1B - MR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.767 | 0.793 | 0.765 | -0.002 | -0.028 | 0.821 | 0.054 | 0.029 |
| 02 | 0.751 | 0.776 | 0.747 | -0.004 | -0.029 | 0.766 | 0.015 | -0.011 |
| 03 | 0.927 | 0.948 | 0.877 | -0.050 | -0.071 | 0.939 | 0.012 | -0.009 |
| 04 | 1.266 | 1.250 | 1.274 | 0.009 | 0.024 | 1.344 | 0.078 | 0.094 |
| 05 | 1.078 | 1.090 | 1.187 | 0.110 | 0.098 | 1.157 | 0.079 | 0.067 |
| 06 | 0.832 | 0.830 | 0.995 | 0.163 | 0.165 | 0.996 | 0.164 | 0.165 |
| 07 | 0.806 | 0.796 | 0.940 | 0.134 | 0.144 | 0.918 | 0.112 | 0.122 |
| 08 | 1.014 | 1.001 | 1.092 | 0.077 | 0.091 | 1.075 | 0.061 | 0.074 |
| 09 | 0.910 | 0.882 | 0.987 | 0.078 | 0.105 | 0.952 | 0.043 | 0.071 |
| 10 | 0.841 | 0.809 | 0.950 | 0.109 | 0.141 | 0.897 | 0.056 | 0.089 |

## Readout

- PL1D beats pTVReg on native snap TRE in `3/10` cases and beats MR-D-BSCP `w=0.055` in `3/10` cases.
- PL1B beats pTVReg on native snap TRE in `0/10` cases and beats MR-D-BSCP `w=0.055` in `2/10` cases.
- Aggregate native snap TRE: pTVReg `0.9191`, MR-D-BSCP `w=0.055` `0.9175`, PL1D `0.9815`, PL1B `0.9865` mm.
- PL1B has slightly better reg-grid subvoxel TRE than PL1D, but worse native snap TRE and higher foreground SDlogJ/fold percentage.
- This remains a protocol-tagged comparison, not a clean method-only claim, because crop, mask, interpolation, and regularization conventions differ across historical rows.

## Source Files

- PL1D/PL1B: `experiments/2026-08-07_lungct_dual_inr_1mm_lungmask_revision/optimization_index.csv`
- pTVReg: `experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv`
- MR-D-BSCP pTV-loss `w=0.055`: `experiments/2026-06-25_lungct_ptvprep_reg_sweep_idir_directcp32/directcp_refined_case_rows.csv`
