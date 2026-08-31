---
experiment_id: 2026-08-30_copd_ptvreg_matlab_vs_mrdbscp_p2e
status: completed_artifact_comparison
dataset: copd_ptv_landmark_crop_1mm
---

# COPD 1-mm pTVReg MATLAB vs MR-D-BSCP P2e

This review folder compares the completed all-ten-case COPD 1-mm pTVReg MATLAB
reproduction with the fixed all-ten-case MR-D-BSCP P2e reference.  It does not
use the separate 4DCT `w=0.055` cohort.

The headline metric is native-domain, snap-to-voxel landmark TRE in mm for
`eBHCT -> iBHCT`.  Both methods provide 300 errors for each COPD case, enabling
casewise empirical CDFs.

- [Casewise report](casewise_ptvreg_vs_mr_d_bscp_p2e_report.md)
- [Casewise values](casewise_ptvreg_vs_mr_d_bscp_p2e.csv)
- [CDF review notebook](copd_ptvreg_vs_mr_d_bscp_p2e_tre_cdfs.ipynb)
- [Displacement-magnitude CDF notebook](copd_ptvreg_vs_mr_d_bscp_p2e_displacement_magnitude_cdfs.ipynb)

## Headline TRE by case

Each case entry is its mean ± sample SD TRE (mm) across the 300 native-domain,
snap-to-voxel landmarks. The final row is the unweighted mean ± sample SD of
the ten case means; it is not the pooled 3,000-landmark statistic.

| COPD case | Challenge case | pTVReg MATLAB TRE (mm) | MR-D-BSCP P2e TRE (mm) |
| --- | ---: | ---: | ---: |
| 01 | 104 | 0.7130 ± 0.8127 | 0.7141 ± 0.8340 |
| 02 | 105 | 1.8987 ± 3.6374 | 3.3480 ± 5.5756 |
| 03 | 106 | 0.7744 ± 0.7734 | 0.8232 ± 0.7868 |
| 04 | 107 | 0.6754 ± 0.7371 | 0.7325 ± 0.9074 |
| 05 | 108 | 0.7131 ± 0.8253 | 0.7404 ± 1.0838 |
| 06 | 109 | 0.6622 ± 1.1533 | 0.8511 ± 1.4799 |
| 07 | 110 | 0.7542 ± 0.8773 | 0.8362 ± 1.0649 |
| 08 | 111 | 0.7810 ± 1.5918 | 0.8196 ± 1.2626 |
| 09 | 112 | 0.6390 ± 1.0147 | 0.8512 ± 1.9594 |
| 10 | 113 | 0.8498 ± 0.8633 | 0.7886 ± 0.7683 |
| **Ten-case mean ± SD** | — | **0.8461 ± 0.3752** | **1.0505 ± 0.8088** |

## Casewise TRE distribution and threshold detail

Every cell below reports **pTVReg MATLAB / MR-D-BSCP P2e** from the same 300
headline-TRE landmarks for that case. Percentiles and maximum are in mm.

| COPD case | Challenge case | p50 | p75 | p90 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 104 | 0.625 / 0.625 | 0.884 / 0.884 | 2.004 / 1.435 | 2.864 / 3.134 | 6.374 / 5.896 |
| 02 | 105 | 0.912 / 0.912 | 2.040 / 2.661 | 3.227 / 12.915 | 20.010 / 21.397 | 30.447 / 31.026 |
| 03 | 106 | 0.652 / 0.652 | 0.922 / 0.922 | 2.500 / 2.500 | 2.820 / 2.894 | 3.240 / 2.894 |
| 04 | 107 | 0.590 / 0.590 | 0.834 / 0.834 | 1.319 / 1.892 | 2.827 / 2.829 | 5.069 / 9.215 |
| 05 | 108 | 0.647 / 0.647 | 0.915 / 0.915 | 1.952 / 2.500 | 2.888 / 2.896 | 6.493 / 11.445 |
| 06 | 109 | 0.633 / 0.633 | 0.633 / 0.895 | 0.895 / 1.415 | 2.691 / 9.184 | 13.722 / 11.672 |
| 07 | 110 | 0.625 / 0.625 | 0.884 / 0.884 | 1.398 / 2.577 | 3.147 / 5.193 | 6.187 / 11.075 |
| 08 | 111 | 0.586 / 0.586 | 0.829 / 0.829 | 1.310 / 1.429 | 3.275 / 3.874 | 24.792 / 16.166 |
| 09 | 112 | 0.664 / 0.664 | 0.664 / 0.664 | 0.939 / 1.328 | 3.727 / 13.577 | 11.732 / 16.421 |
| 10 | 113 | 0.742 / 0.742 | 1.049 / 1.049 | 2.500 / 2.238 | 3.060 / 3.001 | 5.815 / 3.710 |

The next table gives the strict-success ratio (`TRE < threshold`) as a
percentage of the 300 landmarks; each cell remains **pTVReg / P2e**.

| COPD case | Challenge case | <1 mm | <3 mm | <5 mm | <10 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 104 | 85.7 / 85.3% | 99.3 / 98.7% | 99.7 / 99.7% | 100.0 / 100.0% |
| 02 | 105 | 63.7 / 55.0% | 88.0 / 79.0% | 94.3 / 84.3% | 95.7 / 87.3% |
| 03 | 106 | 85.0 / 83.0% | 99.3 / 100.0% | 100.0 / 100.0% | 100.0 / 100.0% |
| 04 | 107 | 84.3 / 82.0% | 99.7 / 99.0% | 99.7 / 99.3% | 100.0 / 100.0% |
| 05 | 108 | 87.0 / 86.3% | 99.3 / 99.0% | 99.7 / 99.3% | 100.0 / 99.7% |
| 06 | 109 | 92.0 / 88.3% | 99.0 / 96.3% | 99.0 / 97.7% | 99.3 / 99.3% |
| 07 | 110 | 83.0 / 79.7% | 98.7 / 98.7% | 99.0 / 98.7% | 100.0 / 99.7% |
| 08 | 111 | 83.7 / 83.0% | 98.3 / 98.3% | 99.3 / 99.0% | 99.7 / 99.7% |
| 09 | 112 | 91.3 / 88.0% | 98.7 / 97.0% | 99.0 / 97.7% | 99.7 / 98.3% |
| 10 | 113 | 71.0 / 72.0% | 98.3 / 98.7% | 99.7 / 100.0% | 100.0 / 100.0% |

## Method labels

- **pTVReg MATLAB reproduction:** official COPD 1-mm path with exact local CC,
  isoTV `0.11`, border mask `5`, linear control-point refinement `8 -> 4`, and
  cubic image interpolation.
- **MR-D-BSCP P2e:** DirectCP cubic B-spline control optimization without an
  INR; control stages `64 -> 32 -> 16 -> 8 -> 4`, image scales
  `16 -> 8 -> 4 -> 2 -> 1`, stage epochs `400,400,400,400,900`, Adam
  `lr=1e-3`, and seed `1`.

The comparison is informative at the matched COPD 1-mm headline metric, but is
not a method-only ablation: control basis, optimizer, multiresolution path, and
image-resampling implementation differ.

## Displacement-magnitude review

The magnitude notebook evaluates and displays the fixed-iBH lung only. This
canonical pTV-style mask is metric/visualization-only: it was not used in either
registration objective. It matches every saved P2e 1-mm field grid exactly; for
the MATLAB `Tptv` fields with a different z size, the notebook makes its
nearest-neighbor mask resampling explicit. Coronal panels follow the reference
notebook's radiology orientation and show the fixed-lung contour. Specifically,
they use the reference lineage's documented raw COPD transform
`z' = Nz - 1 - z` before converting `z,y,x` to display-oriented `x,y,z`.
