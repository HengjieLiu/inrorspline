# COPD DUAL-INR six-arm camera-ready readout

## Cohort identity

- **Dataset:** DIR-Lab COPD cases 01-10, not the separate DIR-Lab 4DCT T00/T50 cohort.
- **Direction:** moving eBHCT to fixed iBHCT (eBH -> iBH), with 300 COPD landmarks per case.
- **Source convention:** native records use the COPD iBH/eBH image-pair layout; P records use the COPD legacy Python-bounds pTV-style 1 mm artifact.

All six new arms use one seed and all ten COPD cases. TRE values are physical mm; pTVReg is intentionally TRE-only because its official artifact has no dense field.

## Casewise headline native/crop snap TRE

| COPD case | A0 | A1 | N0 | N1 | P0 | P1 | pTVReg official | MR-D-BSCP P2e |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.0942 | 1.0747 | 1.2764 | 1.2138 | 1.4468 | 1.4889 | 0.7130 | 0.7141 |
| 2 | 2.1442 | 1.9401 | 2.1140 | 2.1634 | 2.8212 | 2.9213 | 1.9053 | 3.3480 |
| 3 | 1.1139 | 1.0866 | 1.1626 | 1.1725 | 1.3346 | 1.3775 | 0.7735 | 0.8232 |
| 4 | 1.1526 | 1.1068 | 4.7364 | 1.3736 | 1.5839 | 1.6451 | 0.6727 | 0.7325 |
| 5 | 1.0600 | 1.0588 | 1.3250 | 1.2797 | 1.6159 | 1.6522 | 0.7109 | 0.7404 |
| 6 | 1.5045 | 1.4748 | 1.6857 | 3.5436 | 3.8996 | 4.6347 | 0.6622 | 0.8511 |
| 7 | 1.0387 | 1.0791 | 1.1643 | 1.1754 | 1.3215 | 1.3407 | 0.7491 | 0.8362 |
| 8 | 1.2980 | 1.3252 | 1.4590 | 1.4827 | 2.0225 | 2.1246 | 0.7823 | 0.8196 |
| 9 | 0.9817 | 1.0141 | 1.0883 | 1.0632 | 1.2906 | 1.3201 | 0.6412 | 0.8512 |
| 10 | 1.3249 | 1.2828 | 1.5674 | 1.6149 | 1.7298 | 1.7710 | 0.8509 | 0.7886 |

## Population summary

| Method | TRE mean ± sample SD (mm) | SDlogJ | J≤0 (%) | Bending | Diffusion |
|---|---:|---:|---:|---:|---:|
| A0 | 1.2713 ± 0.3453 | 0.3533 ± 0.1691 | 0.2654 ± 0.4579 | 0.0167 ± 0.0077 | 0.0772 ± 0.0336 |
| A1 | 1.2443 ± 0.2853 | 0.3555 ± 0.1721 | 0.2512 ± 0.4330 | 0.0166 ± 0.0081 | 0.0767 ± 0.0345 |
| N0 | 1.7579 ± 1.0906 | 0.3160 ± 0.1380 | 0.0787 ± 0.1472 | 0.0197 ± 0.0094 | 0.0784 ± 0.0358 |
| N1 | 1.6083 ± 0.7500 | 0.3188 ± 0.1581 | 0.1205 ± 0.2129 | 0.0187 ± 0.0097 | 0.0807 ± 0.0366 |
| P0 | 1.9066 ± 0.8368 | 0.3296 ± 0.1636 | 0.0941 ± 0.1794 | 0.0006 ± 0.0002 | 0.0260 ± 0.0102 |
| P1 | 2.0276 ± 1.0351 | 0.3355 ± 0.1662 | 0.1107 ± 0.2383 | 0.0006 ± 0.0002 | 0.0271 ± 0.0108 |
| pTVReg official | 0.8461 ± 0.3775 | N/A | N/A | N/A | N/A |
| MR-D-BSCP P2e | 1.0505 ± 0.8088 | 0.4021 ± 0.1152 | 0.7251 ± 0.7505 | 0.0089 ± 0.0042 | 0.0389 ± 0.0144 |

## Protocol limits

- A0/A1 retain the author native TotalSegmentator five-lobe crop and sampling. A1 removes only the semantic loss and replays its paired A0 blur transitions.
- N0/N1 use native full-volume fixed-iBH R231 sampling. P0/P1 use the existing legacy Python pTV-style 1 mm crop and no lung mask in optimization. Only N0–N1 and P0–P1 are schedule-controlled contrasts.
- Official pTVReg uses exact MATLAB crop bounds and is a contextual TRE-only reference. MR-D-BSCP P2e is the single lowest population-TRE completed 1 mm COPD cohort, used whole rather than per-case selection.
- Diffusion and bending are voxel-coordinate finite-difference diagnostics; cross-domain magnitude comparisons remain protocol-tagged. SDlogJ and folding use the available fixed-iBH R231 metric ROI, and A rows report its crop coverage.

Technical acceptance: 60/60 new records complete; regularity computed for 70 new/reference-field records.
