---
report_type: completed_artifact_comparison
status: review_ready_with_unselected_copd_candidate
datasets:
  - DIR-Lab COPD10 pTV-style 1-mm landmark crop
  - DIR-Lab 4DCT10 pTV-style 1-mm landmark crop
methods:
  - pTVReg MATLAB
  - MR-D-BSCP
metric: native-domain snap-to-voxel landmark TRE (mm)
---

# pTVReg vs MR-D-BSCP: COPD10 and 4DCT10 1-mm TRE summary

## Scope

This report collects the completed, whole-cohort 1-mm pTVReg and MR-D-BSCP
artifacts for the two DIR-Lab datasets.  It is a readout of saved results: no
registration was rerun for this report and no casewise selection was
performed.  The COPD section now also records one completed, explicitly
unselected `w=0.055` transfer candidate alongside the frozen P2e `w=0.11`
reference; it does not replace that reference.

The headline endpoint throughout is **native-domain, snap-to-voxel landmark
TRE in mm**.  COPD uses `eBHCT -> iBHCT`; 4DCT uses `T50 -> T00`.

For every case, the displayed value is **mean +/- sample SD across its 300
landmark TREs**.  Each final dataset row is the **unweighted mean +/- sample
SD of the ten case means**.  It is therefore not a pooled 3,000-landmark
statistic.

## Aggregate result

`Primary - pTVReg` is the frozen MR-D-BSCP protocol minus pTVReg, using the
ten case means.  `Candidate - frozen` is the per-case paired difference of the
completed COPD `w=0.055` transfer candidate from frozen P2e `w=0.11`; its SD
is the sample SD of those ten paired differences.  Negative values favor the
corresponding MR-D-BSCP column on this endpoint.

| Dataset / direction | pTVReg (mm) | Frozen MR-D-BSCP (mm) | Primary - pTVReg (mm) | COPD `w=0.055` candidate (mm) | Candidate - frozen (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| COPD10 1-mm, `eBH -> iBH` | **0.8461 +/- 0.3752** | **P2e `w=0.11`: 1.0505 +/- 0.8088** | **+0.2044** | **P2e geometry `w=0.055`: 1.0501 +/- 0.7331** | **-0.0004 +/- 0.1325** |
| 4DCT10 1-mm, `T50 -> T00` | **0.9191 +/- 0.1607** | **camera-ready `w=0.055`: 0.9168 +/- 0.1561** | **-0.0023** | -- | -- |

The frozen COPD P2e and camera-ready 4DCT rows are separate fixed,
cohort-level protocols.  The COPD candidate is a separately named one-factor
transfer screen with the P2e geometry/schedule held fixed; it is not a new
selected protocol and not the 4DCT camera-ready row.

## Casewise results: COPD10

COPD cases map to challenge cases `104` through `113`.  The frozen MR-D-BSCP
result is P2e, the completed five-stage `64 -> 32 -> 16 -> 8 -> 4` cohort.
The adjacent candidate differs only in isoTV weight (`0.11 -> 0.055`), and was
run across all ten cases without casewise selection.

| COPD case | Challenge case | pTVReg mean +/- SD (mm) | Frozen P2e `w=0.11` mean +/- SD (mm) | Primary - pTVReg (mm) | P2e geometry `w=0.055` candidate mean +/- SD (mm) | Candidate - frozen (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 104 | 0.7130 +/- 0.8127 | 0.7141 +/- 0.8340 | +0.0010 | 0.7343 +/- 0.8667 | +0.0203 |
| 02 | 105 | 1.8987 +/- 3.6374 | 3.3480 +/- 5.5756 | +1.4493 | 3.1162 +/- 5.3751 | -0.2318 |
| 03 | 106 | 0.7744 +/- 0.7734 | 0.8232 +/- 0.7868 | +0.0488 | 0.8407 +/- 0.8064 | +0.0175 |
| 04 | 107 | 0.6754 +/- 0.7371 | 0.7325 +/- 0.9074 | +0.0570 | 0.7986 +/- 1.0125 | +0.0661 |
| 05 | 108 | 0.7131 +/- 0.8253 | 0.7404 +/- 1.0838 | +0.0273 | 0.7671 +/- 1.0608 | +0.0267 |
| 06 | 109 | 0.6622 +/- 1.1533 | 0.8511 +/- 1.4799 | +0.1889 | 0.9279 +/- 1.7268 | +0.0768 |
| 07 | 110 | 0.7542 +/- 0.8773 | 0.8362 +/- 1.0649 | +0.0820 | 0.7997 +/- 0.9668 | -0.0365 |
| 08 | 111 | 0.7810 +/- 1.5918 | 0.8196 +/- 1.2626 | +0.0385 | 0.9721 +/- 1.7122 | +0.1526 |
| 09 | 112 | 0.6390 +/- 1.0147 | 0.8512 +/- 1.9594 | +0.2122 | 0.6253 +/- 1.0161 | -0.2259 |
| 10 | 113 | 0.8498 +/- 0.8633 | 0.7886 +/- 0.7683 | -0.0612 | 0.9190 +/- 1.4659 | +0.1304 |
| **Ten-case mean +/- SD** | -- | **0.8461 +/- 0.3752** | **1.0505 +/- 0.8088** | **+0.2044** | **1.0501 +/- 0.7331** | **-0.0004 +/- 0.1325** |

## Casewise results: 4DCT10

The pTVReg row is the validated saved MATLAB deformation archive used by the
camera-ready source.  The MR-D-BSCP row is the 2,500-step four-stage pTV-loss
`w=0.055` cohort selected once for all ten cases.

| 4DCT case | pTVReg mean +/- SD (mm) | MR-D-BSCP mean +/- SD (mm) | Delta (mm) |
| ---: | ---: | ---: | ---: |
| 01 | 0.7671 +/- 0.8917 | 0.7920 +/- 0.9059 | +0.0248 |
| 02 | 0.7509 +/- 0.8990 | 0.7804 +/- 0.9014 | +0.0295 |
| 03 | 0.9274 +/- 1.0726 | 0.9401 +/- 1.0567 | +0.0127 |
| 04 | 1.2657 +/- 1.2592 | 1.2501 +/- 1.2544 | -0.0156 |
| 05 | 1.0777 +/- 1.4172 | 1.0934 +/- 1.3934 | +0.0156 |
| 06 | 0.8316 +/- 0.9249 | 0.8270 +/- 0.9601 | -0.0046 |
| 07 | 0.8061 +/- 0.9134 | 0.7974 +/- 0.9261 | -0.0087 |
| 08 | 1.0142 +/- 1.2922 | 0.9975 +/- 1.2953 | -0.0167 |
| 09 | 0.9095 +/- 0.9540 | 0.8849 +/- 0.9339 | -0.0246 |
| 10 | 0.8410 +/- 0.9695 | 0.8053 +/- 0.9915 | -0.0357 |
| **Ten-case mean +/- SD** | **0.9191 +/- 0.1607** | **0.9168 +/- 0.1561** | **-0.0023** |

## Method and protocol identity

### pTVReg MATLAB

Both pTVReg rows use the pTVReg 1-mm landmark-crop convention and a physical
fixed-to-moving pull transform for the stated direction.  The saved fields and
their reported landmark TRE contracts were validated before this readout.

- **COPD reproduction:** MATLAB pTVReg best/fine-tune path, exact local CC
  (`sigma=2.1 mm`), physical isoTV weight `0.11`, `csqrt=5e-3`, five-voxel
  border mask, and linear control-grid refinement `8 -> 4`.  The reported
  COPD configuration uses cubic image interpolation; this does not make its
  control grid cubic.
- **4DCT camera-ready reference:** validated uploaded MATLAB pTVReg 1-mm
  deformation fields.  Their saved physical `[x,y,z]` pull-field convention is
  `fixed T00 -> moving T50`, and recomputed native snap TRE agrees with the
  saved per-case TRE.  This report does not infer unrecorded MATLAB solver
  settings beyond that verified field/crop/TRE contract.

### MR-D-BSCP

MR-D-BSCP is direct optimization of cubic B-spline control points: there is
**no INR**.  At every stage it uses pTV-style local CC (`sigma=2.1 mm`), a
five-voxel border domain, physical control-value isoTV with `csqrt=5e-3`,
Adam (`lr=1e-3`), seed `1`, and exact dyadic knot insertion between control
levels.  No lung mask weights either registration objective.

| Protocol element | COPD10 frozen P2e | COPD10 `w=0.055` candidate | 4DCT10 camera-ready row |
| --- | --- | --- | --- |
| Control-point stages | `64 -> 32 -> 16 -> 8 -> 4` | `64 -> 32 -> 16 -> 8 -> 4` | `32 -> 16 -> 8 -> 4` |
| Image downsampling | `16 -> 8 -> 4 -> 2 -> 1` | `16 -> 8 -> 4 -> 2 -> 1` | `8 -> 4 -> 2 -> 1` |
| Stage epochs | `400,400,400,400,900` | `400,400,400,400,900` | `400,400,800,900` |
| Total epochs | `2500` | `2500` | `2500` |
| isoTV weight | `0.11` | `0.055` | `0.055` |
| Cohort choice | Lowest completed COPD 1-mm P2e setting; once for all cases | Completed post-selection transfer candidate; unselected | Selected once from the 4DCT five-weight sweep; never casewise |

Thus the loss family is shared, but the frozen COPD and 4DCT MR-D-BSCP rows
are not one identical hyperparameter setting applied to two datasets.  Their
control lattice, image pyramid, stage allocation, and isoTV weight differ.
The COPD candidate isolates the isoTV-weight change within the frozen COPD
geometry/schedule.  Raw TRE should not be interpreted as a direct
cross-dataset ranking.

## Why COPD P2e uses `0.11`, not `0.055`

The provenance record supports a specific, limited answer:

1. **`0.11` was the frozen pTVReg-compatible default.**  The pTVReg 1-mm
   4DCT and COPD scripts document isoTV `0.11`; the repo's pTV-style spline
   convention retains `0.11` unless a protocol explicitly declares a weight
   sweep.
2. **P2e is a schedule correction of a pre-existing COPD `w=0.11` cohort.**
   P2d and P2e both use controls `64 -> 32 -> 16 -> 8 -> 4`, the same loss,
   and `w=0.11`.  P2e changed only the epoch allocation from
   `100,200,400,800,1000` to `400,400,400,400,900`; it was not a COPD
   regularization-weight screen.
3. **`0.055` was selected on a separate 4DCT screen.**  At the 2,500-step
   4DCT schedule, the completed pTV-loss sweep evaluated
   `0.011, 0.033, 0.055, 0.11, 0.33`.  Their mean native snap TREs were
   `0.9776, 0.9355, 0.9168, 0.9238, 0.9516 mm`, respectively.  `0.055` was
   therefore the lowest-TRE completed **4DCT** weight and was then fixed for
   the camera-ready all-ten-case row.
4. **A completed all-ten-case COPD `w=0.055` transfer candidate is now
   available, but it is not a new selected reference.**  With the P2e geometry
   and schedule frozen, it gives `1.0501 +/- 0.7331 mm` versus
   `1.0505 +/- 0.8088 mm` for frozen `w=0.11`; the paired change is
   `-0.0004 +/- 0.1325 mm`.  It is lower in cases `02`, `07`, and `09`, and
   higher in the other seven.  This establishes that the completed transfer
   does not provide a uniform improvement; it does not establish either weight
   as a COPD weight optimum.

The completed `w=0.055` screen is retained here as a separately named,
unselected variant.  Frozen P2e `w=0.11` remains the COPD reference unless an
explicit protocol-selection decision supersedes it.

## Source artifacts

All values above are copied or recomputed from the following review artifacts.
The source paths are relative to this report folder; raw data and large field
files are not included here.  The camera-ready `v013` aggregate source reports
the 4DCT SDs as population SDs (`0.1525` and `0.1481`).  This report instead
uses the stated sample-SD convention (`ddof=1`) consistently with the COPD
table, yielding `0.1607` and `0.1561` when recomputed from the ten listed 4DCT
case means.

| Dataset / method | Source artifact |
| --- | --- |
| COPD pTVReg | [`casewise_ptvreg_vs_mr_d_bscp_p2e.csv`](../2026-08-30_copd_ptvreg_matlab_vs_mrdbscp_p2e/casewise_ptvreg_vs_mr_d_bscp_p2e.csv) |
| COPD MR-D-BSCP P2e | [`copd_case_rows.csv`](../2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/copd_case_rows.csv) and its [execution specification](../2026-07-07_copd_directcp64_32_16_8_4_ptvlcc_w0p11_e400x4_900_sanity/codex_execution_spec.md) |
| COPD MR-D-BSCP P2e-geometry `w=0.055` candidate | [`candidate comparison`](../2026-08-31_copd_p2e_geometry_w0p055_all10/candidate_vs_p2e_review.md), [`case rows`](../2026-08-31_copd_p2e_geometry_w0p055_all10/copd_case_rows.csv), and its [execution specification](../2026-08-31_copd_p2e_geometry_w0p055_all10/codex_execution_spec.md) |
| 4DCT pTVReg | [`combined_case_rows_with_ptvreg.csv`](../2026-06-26_lungct_directcp4stage_midratio_pareto_extension/pareto_visual_review_midratio_extension/combined_case_rows_with_ptvreg.csv) |
| 4DCT MR-D-BSCP | [`case_rows.csv`](../2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/case_rows.csv), [`mean_rows.csv`](../2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/mean_rows.csv), and its [weight-sweep specification](../2026-08-13_lungct_fig2_mrdbscp_2500_schedule_refresh/codex_execution_spec.md) |
| Camera-ready 4DCT source selection | [`v013 source rows`](../paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/source_rows.csv) and [review report](../paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels/review_report.md) |

## Interpretation boundary

Within a dataset, the tables compare the completed artifacts under the same
headline metric.  The COPD `w=0.055` row is a one-factor regularization
transfer screen, not a selected replacement or a method-only result.  pTVReg
and MR-D-BSCP differ in control basis, optimizer, multiresolution
implementation, and image-resampling details.  The two datasets also have
different motion regimes and MR-D-BSCP operating points.  These results should
therefore be read as protocol-labelled empirical comparisons, not as an
INR-versus-spline or cross-dataset causal claim.
