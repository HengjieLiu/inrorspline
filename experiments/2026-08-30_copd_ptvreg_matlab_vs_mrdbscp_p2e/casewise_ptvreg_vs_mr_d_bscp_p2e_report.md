---
doc_type: result_comparison
status: completed_artifact_readout
---

# COPD 1-mm pTVReg MATLAB vs MR-D-BSCP P2e

## Scope and metric

This report compares the completed all-ten-case MATLAB pTVReg COPD rerun to
MR-D-BSCP P2e, the fixed completed 1-mm DirectCP reference.  It does **not**
use the 4DCT `w=0.055` MR-D-BSCP cohort.

Both rows use `eBHCT -> iBHCT` and headline native-domain snap-to-voxel TRE in
mm.  For each case, pTVReg's saved MATLAB `TRE_all_phys` vector and P2e's
`per_landmark_tre_native_crop.csv` `snap_tre_mm` column provide 300 finite
landmark errors.  This makes their CDFs directly inspectable under the stated
metric convention.

| Method | Cases | Mean case TRE +/- sample SD (mm) | Mean P2e - pTVReg (mm) |
| --- | ---: | ---: | ---: |
| pTVReg MATLAB reproduction | 10 | 0.8461 +/- 0.3752 | -- |
| MR-D-BSCP P2e | 10 | 1.0505 +/- 0.8088 | +0.2044 |

pTVReg has the lower case-mean TRE in nine of ten cases; P2e is lower only for
COPD10 / challenge case 113.

## Casewise native snap TRE

Positive delta means P2e has higher mean TRE.  The `>5 mm` columns count
landmarks in each 300-landmark CDF tail.

| COPD case | Challenge case | pTVReg mean | P2e mean | Delta | pTVReg p95 | P2e p95 | pTVReg >5 mm | P2e >5 mm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 104 | 0.713 | 0.714 | +0.001 | 2.577 | 2.581 | 1 | 1 |
| 02 | 105 | 1.899 | 3.348 | +1.449 | 5.690 | 17.789 | 17 | 47 |
| 03 | 106 | 0.774 | 0.823 | +0.049 | 2.584 | 2.588 | 0 | 0 |
| 04 | 107 | 0.675 | 0.732 | +0.057 | 2.569 | 2.569 | 1 | 1 |
| 05 | 108 | 0.713 | 0.740 | +0.027 | 2.582 | 2.586 | 1 | 2 |
| 06 | 109 | 0.662 | 0.851 | +0.189 | 1.415 | 2.579 | 3 | 7 |
| 07 | 110 | 0.754 | 0.836 | +0.082 | 2.577 | 2.581 | 3 | 4 |
| 08 | 111 | 0.781 | 0.820 | +0.039 | 2.568 | 2.571 | 2 | 3 |
| 09 | 112 | 0.639 | 0.851 | +0.212 | 1.904 | 2.587 | 3 | 7 |
| 10 | 113 | 0.850 | 0.789 | -0.061 | 2.608 | 2.608 | 1 | 0 |

## COPD2 diagnostic

COPD2 is not a uniform offset failure.  The two methods have the same median
TRE, `0.912 mm`, but P2e has a heavier tail: its 75th percentile is `2.661 mm`
versus `2.040 mm`, its 95th percentile is `17.789 mm` versus `5.690 mm`, and
it has `47/300` errors above `5 mm` versus `17/300` for pTVReg.  The CDF
notebook therefore focuses attention on the upper tail and preserves landmark
order for paired follow-up.

This identifies where to inspect next, rather than establishing a cause.  The
next audit should locate the COPD2 landmark indices where P2e has large paired
errors, map them to anatomy and crop-border distance, and compare local
displacements against the saved MATLAB field.

## Protocol guardrail

pTVReg uses its MATLAB crop/resampling and linear control-point implementation;
P2e uses direct cubic B-spline controls, exact dyadic knot insertion, and Adam.
Thus this is a matched-dataset, matched-headline-metric comparison, not proof
that the representation alone accounts for the observed differences.

## Artifact inputs

The notebook intentionally resolves these external inputs through environment
variables, keeping host paths out of review-facing files:

- `PTVREG_COPD_MATLAB_ROOT`: the completed GPU MATLAB
  `copd_published_1mm_cp1_exactlcc_cubic` case directory.
- `MRDBSCP_P2E_COPD_ROOT`: the completed P2e method directory containing the
  ten `per_landmark_tre_native_crop.csv` files.
