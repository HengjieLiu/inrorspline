---
experiment_id: 2026-08-31_copd1mm_ptvreg_vs_mrdbscp_p2e_multislice_visual_audit
status: posthoc_visualization_audit
dataset: copd_ptv_landmark_crop_1mm
---

# COPD 1-mm pTVReg vs MR-D-BSCP P2e multi-slice visual audit

This folder is a post-registration visual audit of the completed all-ten-case
COPD 1-mm pTVReg MATLAB reproduction and the frozen all-ten-case MR-D-BSCP P2e
reference. It does not launch registration or introduce a new numerical result.

The review notebook places both reconstructions on the canonical P2e fixed-iBH
1-mm grid. The pTVReg archive provides a native full-grid physical pull field
(`Tptv_rsz`, components `x,y,z`); it is sampled at the native coordinates of
that common grid before sampling the moving eBH image or its lung mask. P2e's
saved dense field is already a fixed-grid pull field in `z,y,x` voxel order on
the canonical 1-mm grid. At 1-mm isotropic spacing its magnitude is in mm.

## Review surface

[`copd1mm_ptvreg_vs_mrdbscp_p2e_multislice_review.ipynb`](copd1mm_ptvreg_vs_mrdbscp_p2e_multislice_review.ipynb)
contains executed inline figures for COPD cases 01–10 (challenge cases 104–113).
For every case, five fixed-lung AP locations are selected from the occupied
fixed-mask y range at 15%, 35%, 50%, 65%, and 85%.

Each row contains moving eBH, fixed iBH, both deformed moving images, both
`deformed - fixed` residuals, and both full-grid displacement magnitudes.
Residuals use a zero-centred symmetric `bwr` scale. Both residual and magnitude
scales are shared across the two methods and all five slices within a case.

Fixed, original-moving, pTVReg-warped-moving, and P2e-warped-moving lung
boundaries are overlaid on every panel. The masks are visualization-only; they
were not used by either registration objective.

## Display contract

The stored COPD arrays are `z,y,x`. The notebook applies the established raw
COPD transform—reverse z, convert to `x,y,z`, then flip left/right for the
radiology display—before plotting each coronal y plane. Display markers are
`S` top, `I` bottom, `R` left, and `L` right. AP is the slice-normal axis and is
reported by the fixed-lung y-range quantile and index; the notebook deliberately
does not assign an unsupported anterior/posterior sign to increasing raw y.

## Interpretation boundary

The two registrations retain differences in control basis, optimizer,
multiresolution path, and resampling implementation. This is therefore a
common-grid diagnostic, not a method-only ablation or a replacement for the
native-crop snap-to-voxel TRE comparison in
[`../2026-08-30_copd_ptvreg_matlab_vs_mrdbscp_p2e/`](../2026-08-30_copd_ptvreg_matlab_vs_mrdbscp_p2e/).
