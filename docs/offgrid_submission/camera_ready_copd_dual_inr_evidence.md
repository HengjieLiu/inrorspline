# Camera-ready evidence note: COPD DUAL-INR six-arm baseline

**Status:** complete technical cohort; review before changing manuscript claims.

All six newly implemented DUAL-INR arms completed for **DIR-Lab COPD** cases
01--10—not the separate DIR-Lab 4DCT T00/T50 cohort—with seed 42 and
eBH-to-iBH registration. The report is descriptive: the
arms deliberately retain different preprocessing and optimization domains, so
only the paired schedule contrasts N0--N1 and P0--P1 isolate the
MR-D-BSCP-shaped image-pyramid schedule.

| Method | Preprocessing | Headline physical TRE, mean +/- sample SD (mm) | SD(log-J) | J<=0 (%) |
|---|---|---:|---:|---:|
| A0 | author native five-lobe crop; released semantic loss | 1.2713 +/- 0.3453 | 0.3533 | 0.2654 |
| A1 | A0 data and trace; semantic term removed | 1.2443 +/- 0.2853 | 0.3555 | 0.2512 |
| N0 | native full volume; fixed-iBH R231 sampling | 1.7579 +/- 1.0906 | 0.3160 | 0.0787 |
| N1 | N0 data; fixed [8,4,2,1] pyramid | 1.6083 +/- 0.7500 | 0.3188 | 0.1205 |
| P0 | legacy Python-bounds pTV-style 1 mm crop; no lung mask | 1.9066 +/- 0.8368 | 0.3296 | 0.0941 |
| P1 | P0 data; fixed [8,4,2,1] pyramid | 2.0276 +/- 1.0351 | 0.3355 | 0.1107 |
| official pTVReg | official MATLAB-crop 1 mm reference | 0.8461 +/- 0.3775 | N/A | N/A |
| MR-D-BSCP P2e | predeclared whole 1 mm cohort reference | 1.0505 +/- 0.8088 | 0.4021 | 0.7251 |

The official pTVReg row is TRE-only because no official dense field was
available. MR-D-BSCP P2e is the single completed 1 mm cohort with the lowest
population native-snap TRE; it is not selected per case. For A/N/P and P2e,
regularity uses the fixed-iBH R231 ROI mapped into the method's valid
fixed-space domain. Diffusion and bending values are retained in the detailed
report but remain voxel-grid diagnostics and must not be compared as a
method-only quantity across native and 1 mm protocols.

Detailed casewise TRE conventions, coverage, SD(log-J), folding, bending, and
diffusion values are in
`experiments/2026-08-03_dual_inr_copd_six_arm/camera_ready_readout.md` and
`casewise_tre_and_regularity.csv`.
