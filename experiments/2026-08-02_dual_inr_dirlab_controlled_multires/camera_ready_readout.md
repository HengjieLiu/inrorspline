# Camera-ready controlled DUAL-INR readout

Optimization acceptance: **pass** (40/40). All values below are descriptive; no performance threshold was used.

## Headline/native-domain TRE

Across-case mean ± sample SD in mm. Sample quartiles and IQR are retained in `population_statistics.csv/json`.

| Variant/reference | Preprocessing | Headline snap TRE | Headline subvoxel TRE |
|---|---|---:|---:|
| N0 | Native full volume, native anisotropic spacing, no crop/resampling; fixed/T00 R231 lung mask is the optimization sampling domain. | 1.0738 ± 0.2041 | 1.1739 ± 0.1445 |
| N1 | Native full volume, native anisotropic spacing, no crop/resampling; fixed/T00 R231 lung mask is the optimization sampling domain. | 1.0649 ± 0.2091 | 1.1759 ± 0.1462 |
| P0 | Canonical pTVReg-style landmark-union crop at isotropic 1 mm, CT [80,900] mapped to [0,1], five-voxel full-grid border exclusion, and no lung mask in optimization. | 1.0410 ± 0.2160 | 1.1710 ± 0.1670 |
| P1 | Canonical pTVReg-style landmark-union crop at isotropic 1 mm, CT [80,900] mapped to [0,1], five-voxel full-grid border exclusion, and no lung mask in optimization. | 1.0461 ± 0.2211 | 1.1735 ± 0.1692 |
| pTVReg canonical 1 mm (reference) | Validated uploaded pTVReg field on the fixed/moving landmark-union crop ([x,y,z] margin [10,10,5]), CT [80,900] to [0,1], approximately isotropic 1 mm; no lung mask in optimization and fixed/T00 lung mask for regularity evaluation only. | 0.9191 ± 0.1607 | 1.0553 ± 0.1259 |
| MR-D-BSCP best 1 mm (reference) | Our PyTorch pTV-style landmark crop and isotropic-1-mm path with no lung mask in optimization, a 32-16-8-4 control schedule, 8-4-2-1 image schedule, local NCC, and isoTV; fixed/T00 lung mask is used for regularity evaluation only. | 0.9175 ± 0.1561 | 1.0478 ± 0.1306 |

## Registration-grid TRE diagnostics

| Variant/reference | Grid snap TRE (mm) | Grid subvoxel TRE (mm) | Direct-INR snap TRE (mm) |
|---|---:|---:|---:|
| N0 | 1.0738 ± 0.2041 | 1.1739 ± 0.1445 | 1.0738 ± 0.2041 |
| N1 | 1.0649 ± 0.2091 | 1.1759 ± 0.1462 | 1.0649 ± 0.2091 |
| P0 | 1.2766 ± 0.1562 | 1.1780 ± 0.1692 | 1.2741 ± 0.1556 |
| P1 | 1.2773 ± 0.1537 | 1.1805 ± 0.1715 | 1.2758 ± 0.1572 |
| pTVReg canonical 1 mm (reference) | 1.1676 ± 0.1187 | 1.0574 ± 0.1230 | NA |
| MR-D-BSCP best 1 mm (reference) | 1.1618 ± 0.1200 | 1.0493 ± 0.1290 | NA |

## Case-by-case headline snap TRE

| Case | N0 native | N1 native MR-shaped | P0 pTV 1 mm | P1 pTV 1 mm MR-shaped | pTVReg canonical 1 mm* | MR-D-BSCP w=0.055 1 mm* |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.8002 | 0.7735 | 0.7110 | 0.7042 | 0.7671 | 0.7926 |
| 02 | 0.7555 | 0.7583 | 0.7188 | 0.7258 | 0.7509 | 0.7765 |
| 03 | 0.8585 | 0.8425 | 0.8955 | 0.9033 | 0.9274 | 0.9484 |
| 04 | 1.3317 | 1.3401 | 1.2928 | 1.2982 | 1.2657 | 1.2501 |
| 05 | 1.2846 | 1.2740 | 1.3447 | 1.3785 | 1.0777 | 1.0897 |
| 06 | 1.1388 | 1.1289 | 1.0632 | 1.0558 | 0.8316 | 0.8303 |
| 07 | 1.0975 | 1.0651 | 1.1393 | 1.1178 | 0.8061 | 0.7960 |
| 08 | 1.2399 | 1.2368 | 1.1800 | 1.1890 | 1.0142 | 1.0008 |
| 09 | 1.1688 | 1.1746 | 1.0490 | 1.0612 | 0.9095 | 0.8817 |
| 10 | 1.0625 | 1.0556 | 1.0158 | 1.0275 | 0.8410 | 0.8085 |

### Reference selection and preprocessing

- pTVReg tag: `ptvreg_uploaded_field+canonical_ptv_crop_iso1mm+metric_only_fixed_t00_mask`. Validated uploaded pTVReg field on the fixed/moving landmark-union crop ([x,y,z] margin [10,10,5]), CT [80,900] to [0,1], approximately isotropic 1 mm; no lung mask in optimization and fixed/T00 lung mask for regularity evaluation only. Source: `${REPO_ROOT}/experiments/2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv`.
- MR-D-BSCP tag: `mr_d_bscp_32_16_8_4+ptv_crop_iso1mm+ptv_lcc_isotv+metric_only_fixed_t00_mask`. Our PyTorch pTV-style landmark crop and isotropic-1-mm path with no lung mask in optimization, a 32-16-8-4 control schedule, 8-4-2-1 image schedule, local NCC, and isoTV; fixed/T00 lung mask is used for regularity evaluation only. The fixed reference setting is `w=0.055` for all ten cases, selected by the rule “single cohort-level regularization setting with the lowest ten-case headline/native snap TRE; never per-case selection” (cohort mean TRE 0.9175 mm). There is no per-case oracle selection.

### Case-by-case 1 mm reference TRE diagnostics

| Case | pTVReg native snap | pTVReg native sub | pTVReg grid snap | pTVReg grid sub | MR-D-BSCP w=0.055 native snap | MR-D-BSCP w=0.055 native sub | MR-D-BSCP w=0.055 grid snap | MR-D-BSCP w=0.055 grid sub |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.7671 | 0.9417 | 1.0680 | 0.9436 | 0.7926 | 0.9423 | 1.0765 | 0.9465 |
| 02 | 0.7509 | 0.9320 | 1.0575 | 0.9316 | 0.7765 | 0.9427 | 1.0855 | 0.9443 |
| 03 | 0.9274 | 1.0727 | 1.1658 | 1.0759 | 0.9484 | 1.0695 | 1.1625 | 1.0742 |
| 04 | 1.2657 | 1.3206 | 1.4239 | 1.3123 | 1.2501 | 1.3368 | 1.4326 | 1.3332 |
| 05 | 1.0777 | 1.2073 | 1.2933 | 1.2094 | 1.0897 | 1.2015 | 1.3026 | 1.2044 |
| 06 | 0.8316 | 1.0276 | 1.1498 | 1.0341 | 0.8303 | 1.0223 | 1.1428 | 1.0239 |
| 07 | 0.8061 | 0.9715 | 1.1022 | 0.9793 | 0.7960 | 0.9511 | 1.0863 | 0.9554 |
| 08 | 1.0142 | 1.1089 | 1.2401 | 1.1069 | 1.0008 | 1.0743 | 1.1867 | 1.0691 |
| 09 | 0.9095 | 1.0011 | 1.1040 | 1.0098 | 0.8817 | 0.9862 | 1.0830 | 0.9912 |
| 10 | 0.8410 | 0.9700 | 1.0710 | 0.9707 | 0.8085 | 0.9516 | 1.0597 | 0.9507 |

### Case-by-case 1 mm reference regularity

| Case | pTVReg diffusion FG | pTVReg bending FG | pTVReg SDlogJ FG | pTVReg J≤0 (%) | MR-D-BSCP w=0.055 diffusion FG | MR-D-BSCP w=0.055 bending FG | MR-D-BSCP w=0.055 SDlogJ FG | MR-D-BSCP w=0.055 J≤0 (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.003283 | 0.002556 | 0.105081 | 0.000000 | 0.004439 | 0.001710 | 0.125825 | 0.000000 |
| 02 | 0.005278 | 0.004134 | 0.110857 | 0.000943 | 0.007021 | 0.003027 | 0.142879 | 0.000193 |
| 03 | 0.006045 | 0.004391 | 0.129359 | 0.000509 | 0.007755 | 0.003086 | 0.158123 | 0.002519 |
| 04 | 0.011565 | 0.009243 | 0.230515 | 0.006513 | 0.014136 | 0.005727 | 0.252551 | 0.004469 |
| 05 | 0.012421 | 0.009839 | 0.241539 | 0.092490 | 0.015488 | 0.006546 | 0.275161 | 0.096139 |
| 06 | 0.017258 | 0.015972 | 0.274711 | 0.060836 | 0.020134 | 0.009259 | 0.305612 | 0.093676 |
| 07 | 0.013096 | 0.009966 | 0.211059 | 0.004196 | 0.015811 | 0.006111 | 0.239188 | 0.013752 |
| 08 | 0.019012 | 0.012394 | 0.236579 | 0.062781 | 0.020932 | 0.006831 | 0.258907 | 0.098705 |
| 09 | 0.011769 | 0.009867 | 0.202411 | 0.008193 | 0.014840 | 0.005938 | 0.223013 | 0.012130 |
| 10 | 0.016329 | 0.011195 | 0.235259 | 0.041177 | 0.019537 | 0.006940 | 0.257083 | 0.045275 |

## Paired schedule effects

Deltas are multiresolution minus full-resolution within the same preprocessing protocol.

| Pair | Headline snap delta (mm) | Grid snap delta (mm) | SDlogJ FG delta | J≤0 FG delta (pp) |
|---|---:|---:|---:|---:|
| N1_minus_N0 | -0.0089 ± 0.0134 | -0.0089 ± 0.0134 | 0.0004 ± 0.0006 | 0.0000 ± 0.0000 |
| P1_minus_P0 | 0.0051 ± 0.0147 | 0.0006 ± 0.0073 | 0.0003 ± 0.0003 | 0.0000 ± 0.0000 |

## Common posthoc regularity

| Variant/reference | Diffusion FG | Bending FG | SDlogJ FG | J≤0 FG (%) |
|---|---:|---:|---:|---:|
| N0 | 0.0134 ± 0.0080 | 0.0048 ± 0.0023 | 0.1282 ± 0.0315 | 0.0000 ± 0.0000 |
| N1 | 0.0135 ± 0.0080 | 0.0046 ± 0.0022 | 0.1286 ± 0.0315 | 0.0000 ± 0.0000 |
| P0 | 0.0059 ± 0.0027 | 0.0002 ± 0.0001 | 0.1135 ± 0.0342 | 0.0000 ± 0.0000 |
| P1 | 0.0060 ± 0.0027 | 0.0002 ± 0.0001 | 0.1138 ± 0.0342 | 0.0000 ± 0.0000 |
| pTVReg canonical 1 mm (reference) | 0.0116 ± 0.0053 | 0.0090 ± 0.0041 | 0.1977 ± 0.0604 | 0.0278 ± 0.0338 |
| MR-D-BSCP best 1 mm (reference) | 0.0140 ± 0.0058 | 0.0055 ± 0.0023 | 0.2238 ± 0.0607 | 0.0367 ± 0.0431 |

Diffusion and bending are voxel-coordinate finite-difference diagnostics. Their magnitudes are not physical-coordinate energies and should not be compared directly between native anisotropic N and isotropic-1-mm P/reference preprocessing. Jacobian statistics are dimensionless. P/reference lung masks are metric-only and absent from optimization.

## Runtime and peak memory

| Variant | Training seconds | Total seconds | Peak reserved GiB |
|---|---:|---:|---:|
| N0 | 280.5870 ± 12.5018 | 322.4330 ± 19.1257 | 8.3568 ± 0.1540 |
| N1 | 283.8779 ± 8.3615 | 324.0386 ± 13.4017 | 8.4025 ± 0.1449 |
| P0 | 281.2240 ± 10.5484 | 317.5420 ± 14.4995 | 8.3918 ± 0.0626 |
| P1 | 298.5406 ± 13.6129 | 337.9695 ± 21.6464 | 8.4328 ± 0.0501 |

## Interpretation limits

- N0↔N1 and P0↔P1 isolate the image-pyramid schedule under their respective preprocessing protocols.
- N↔P changes crop, resampling, intensity preprocessing, and sampling domain; it is a protocol comparison, not a clean method-only effect.
- N1/P1 are MR-D-BSCP-shaped (MR-inspired) image pyramids. They are not literal B-spline control-point schedules and use no knot insertion.
- A single frozen DUAL hyperparameter recipe was transferred across protocols. These rows are controlled baselines, not claims of protocol-specific optimal tuning.
- The original/native and pTVReg registration-grid TRE conventions are both retained; only explicitly matched domains should be compared.
- pTVReg and MR-D-BSCP rows are contextual 1 mm references with separately tagged preprocessing. MR-D-BSCP uses one cohort-selected setting for all cases, never case-wise best values.
