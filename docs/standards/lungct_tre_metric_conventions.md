---
doc_type: lungct_tre_metric_conventions
status: active
---

# LungCT TRE Metric Conventions

LungCT/4DCT/COPD landmark TRE is sensitive to both the displacement-field
evaluation mode and the landmark coordinate domain. Every reported row should
state both.

## Metric Domains

Use these labels:

- `native_full_or_crop`: TRE in the original voxel coordinate system and native
  spacing. For pTV-preprocessed runs, the learned 1mm displacement is resized
  back to the native landmark crop or full native volume before landmark TRE.
  This is the literature-comparable pTVReg/DIR-Lab-style domain.
- `cropped_1mm_reg`: TRE directly in the resampled cropped 1mm registration
  grid. Landmarks in this grid are generally fractional because native integer
  landmarks were transformed through a crop and 1mm resampling.

Do not compare `native_full_or_crop` TRE and `cropped_1mm_reg` TRE as if they
were the same metric. A method can train on 1mm images while still reporting
the headline TRE after mapping the dense field back to native/crop coordinates.

## Snap And Subvoxel Definitions

Use these definitions for dense displacement fields:

- `snap_to_voxel`: sample displacement at the rounded fixed landmark location
  and round the predicted moving landmark before computing TRE.
- `subvoxel`: trilinearly sample displacement at the fixed landmark and do not
  round the predicted moving landmark before computing TRE.

In `native_full_or_crop`, DIR-Lab/COPD landmarks are integer voxel annotations,
so `snap_to_voxel` is often smaller than `subvoxel`. In `cropped_1mm_reg`, the
landmarks are usually fractional after resampling, so rounding the predicted
point can add quantization error and `snap_to_voxel` can be larger than
`subvoxel`.

This sign flip is not automatically a registration failure. It is expected when
changing metric domains.

## pTVReg Convention

pTVReg reports LungCT/4DCT TRE in snap-to-voxel fashion:

- `external/pTVreg/examples_ptv/DIR_test_all.m` trains on the cropped domain,
  optionally resamples the images to 1mm, then resizes the displacement field
  back to the original crop/full native grid before calling `DIR_movepoints`.
- `external/pTVreg/mutils/My/DIR_helpers/DIR_movepoints.m` uses
  `simple_move_points`, then rounds the moved landmark before computing TRE.
- `external/pTVreg/DIRandCOPD.md` and `external/pTVreg/Examples.md` state that
  the reported DIR/COPD TRE is snap-to-voxel or snap-to-pixel.

Therefore pTVReg-comparable LungCT headline rows should use
`native_full_or_crop` + `snap_to_voxel`. Keep `subvoxel` and
`cropped_1mm_reg` values as diagnostics unless a table explicitly targets the
1mm registration domain.

## Reporting Rule

For LungCT pTVReg-style experiments, report and keep track of all four TRE
values whenever dense fields and landmarks make them available:

- `original_snap`: native/full-or-crop snap-to-voxel TRE.
- `original_subvoxel`: native/full-or-crop subvoxel TRE.
- `reg1mm_snap`: cropped 1mm registration-grid snap-to-voxel TRE.
- `reg1mm_subvoxel`: cropped 1mm registration-grid subvoxel TRE.

Use `original_snap` as the main reporting and comparison value. The other three
columns are required diagnostics for identifying metric-domain effects,
rounding effects, and protocol mismatches.

For the pTVReg 4DCT resampled run, the important detail is that registration
and TRE are not evaluated in the same coordinate grid:

1. pTVReg crops the original/native fixed and moving images around the union of
   fixed/moving landmarks.
2. If the `resize` option is enabled, pTVReg resamples the cropped images to
   approximately 1mm isotropic spacing and runs registration in that resampled
   crop.
3. The estimated displacement field `Tptv` is then resized back from the 1mm
   registration crop to `bszv`, the original native crop size.
4. The resized displacement field is uncropped back into the original/native
   image grid.
5. TRE is computed on original/native landmark coordinates: displacement is
   sampled at the rounded fixed landmark, the moved landmark is rounded to the
   nearest integer voxel, and physical distance to the integer moving landmark
   is computed with the original native spacing `spc_orig`.

This means a pTVReg row can truthfully be described as using a 1mm resampled
registration image while still using original/native snap-to-voxel TRE as the
headline metric. It should not be compared directly to TRE computed only in the
fractional 1mm registration landmark grid.

## pTV-Prep Crop-Origin Warning

Legacy task4 pTV-prep outputs used a task4-compatible Python crop convention.
The canonical regenerated handle `ptvreg_landmark_crop_1mm_trilinear_v1` uses
the corrected pTVReg/MATLAB-origin crop convention. In the current 4DCT audit,
the legacy crop was shifted by `+1` voxel in z/y/x for all ten cases while
keeping the same crop shape.

This crop-origin difference does not by itself corrupt TRE if images,
landmarks, masks, uncropping, and TRE evaluation all use the same convention.
It is still a protocol difference and must be tagged when comparing old task4
rows to regenerated pTV-prep rows.
